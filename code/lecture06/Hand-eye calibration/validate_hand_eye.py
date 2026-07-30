"""Validation script for lecture 6 hand-eye calibration results."""

from __future__ import annotations

import argparse
from pathlib import Path

from common_transforms import (
    build_pairing_report,
    invert_transform,
    load_pose_records,
    load_transform_from_file,
    pair_pose_records,
    save_json,
    summarize_transform_constancy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a hand-eye calibration transform.")
    parser.add_argument("--transform", required=True, help="JSON file containing the solved 4x4 transform.")
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
    parser.add_argument("--report", required=True, help="Output validation report JSON path.")
    parser.add_argument(
        "--transform-key",
        default="matrix",
        help="Matrix key in the transform JSON file.",
    )
    parser.add_argument(
        "--robot-matrix-key",
        default="T_base_end",
        help="Matrix key inside robot pose records.",
    )
    parser.add_argument(
        "--camera-matrix-key",
        default="T_cam_board",
        help="Matrix key inside camera pose records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transform_payload, T = load_transform_from_file(args.transform, matrix_key=args.transform_key)
    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    pairing_report = build_pairing_report(robot_records, camera_records)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    topology = transform_payload.get("topology", "unknown")
    if topology == "eye_in_hand":
        derived = [robot @ T @ camera for robot, camera in zip(robot_mats, camera_mats)]
        derived_name = "T_base_board"
        chain = "T_base_board = T_base_end · T_end_camera · T_cam_board"
    elif topology == "eye_to_hand":
        derived = [invert_transform(robot) @ T @ camera for robot, camera in zip(robot_mats, camera_mats)]
        derived_name = "T_end_board"
        chain = "T_end_board = inv(T_base_end) · T_base_camera · T_cam_board"
    else:
        raise ValueError(
            "无法从 transform JSON 中识别 topology。请确认求解结果文件包含 eye_in_hand 或 eye_to_hand。"
        )

    constancy = summarize_transform_constancy(sample_ids, derived)
    report = {
        "transform_name": transform_payload.get("transform_name", "unknown"),
        "topology": topology,
        "sample_count": len(sample_ids),
        "pairing": pairing_report,
        "validation_chain": chain,
        "derived_transform_name": derived_name,
        "residual_summary": constancy["summary"],
        "worst_translation_sample": constancy["worst_translation_sample"],
        "worst_rotation_sample": constancy["worst_rotation_sample"],
        "reference_transform": constancy["reference_transform"],
        "per_sample": constancy["per_sample"],
        "sample_ids": sample_ids,
    }

    save_json(args.report, report)

    print("Validation complete.")
    print(f"  topology: {topology}")
    print(f"  chain: {chain}")
    print(f"  samples: {len(sample_ids)}")
    if pairing_report["missing_in_camera"]:
        print(f"  missing in camera: {pairing_report['missing_in_camera']}")
    if pairing_report["missing_in_robot"]:
        print(f"  missing in robot: {pairing_report['missing_in_robot']}")
    print(f"  mean translation residual: {report['residual_summary']['mean_translation_mm']:.3f} mm")
    print(f"  mean rotation residual: {report['residual_summary']['mean_rotation_deg']:.4f} deg")
    print(f"  worst translation sample: {report['worst_translation_sample']['id']}")
    print(f"  worst rotation sample: {report['worst_rotation_sample']['id']}")
    print(f"  report saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
