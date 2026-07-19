"""Validation script for lecture 6 hand-eye calibration results."""

from __future__ import annotations

import argparse
from pathlib import Path

from common_transforms import load_pose_records, pair_pose_records, pose_error, validate_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a hand-eye calibration transform.")
    parser.add_argument("--transform", required=True, help="JSON file containing the solved 4x4 transform.")
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
    parser.add_argument("--report", required=True, help="Output validation report JSON path.")
    parser.add_argument(
        "--transform-key",
        default="matrix",
        help="Matrix key in the transform JSON file.")
    parser.add_argument(
        "--robot-matrix-key",
        default="T_base_end",
        help="Matrix key inside robot pose records.")
    parser.add_argument(
        "--camera-matrix-key",
        default="T_cam_board",
        help="Matrix key inside camera pose records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import json
    import numpy as np

    transform_payload = json.loads(Path(args.transform).read_text(encoding="utf-8"))
    if args.transform_key not in transform_payload:
        raise KeyError(f"{args.transform} 中没有找到 {args.transform_key!r}。")
    T = validate_transform(transform_payload[args.transform_key], "calibration transform")

    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    topology = transform_payload.get("topology", "unknown")
    if topology == "eye_in_hand":
        base_board = [robot @ T @ camera for robot, camera in zip(robot_mats, camera_mats)]
    else:
        base_board = [T @ camera for camera in camera_mats]

    ref = base_board[0]
    errors = [pose_error(ref, current) for current in base_board[1:]]
    trans_mm = np.array([item[0] * 1000.0 for item in errors], dtype=np.float64)
    rot_deg = np.array([item[1] for item in errors], dtype=np.float64)

    report = {
        "transform_name": transform_payload.get("transform_name", "unknown"),
        "topology": topology,
        "sample_count": len(sample_ids),
        "mean_translation_mm": float(trans_mm.mean()) if len(trans_mm) else 0.0,
        "max_translation_mm": float(trans_mm.max()) if len(trans_mm) else 0.0,
        "mean_rotation_deg": float(rot_deg.mean()) if len(rot_deg) else 0.0,
        "max_rotation_deg": float(rot_deg.max()) if len(rot_deg) else 0.0,
        "sample_ids": sample_ids,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Validation complete.")
    print(f"  samples: {len(sample_ids)}")
    print(f"  mean translation residual: {report['mean_translation_mm']:.3f} mm")
    print(f"  mean rotation residual: {report['mean_rotation_deg']:.4f} deg")
    print(f"  report saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
