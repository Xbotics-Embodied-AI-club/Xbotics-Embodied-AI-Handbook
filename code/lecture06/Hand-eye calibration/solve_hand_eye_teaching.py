"""Teaching-oriented hand-eye solver that exposes the AX=XB logic."""

from __future__ import annotations

import argparse
from pathlib import Path

from common_transforms import (
    build_relative_motion_pairs,
    invert_transform,
    load_pose_records,
    matrix_to_list,
    pair_pose_records,
    save_json,
    solve_hand_eye_park,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve hand-eye calibration with a transparent teaching solver.")
    parser.add_argument("--topology", required=True, choices=["eye_in_hand", "eye_to_hand"])
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
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

    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    if args.topology == "eye_in_hand":
        motion_pairs = build_relative_motion_pairs(sample_ids, robot_mats, camera_mats)
        transform_name = "T_end_camera"
        explanation = "使用相对运动对 A X = X B，其中 A 来自 gripper 运动，B 来自 board 在相机中的相对运动。"
    else:
        robot_forward = [invert_transform(T) for T in robot_mats]
        motion_pairs = build_relative_motion_pairs(sample_ids, robot_forward, camera_mats)
        transform_name = "T_base_camera"
        explanation = "Eye-to-Hand 通过将机器人位姿改写为 base<-end 的相对运动后，继续使用 A X = X B 的统一形式。"

    X, diagnostics = solve_hand_eye_park(motion_pairs)
    payload = {
        "topology": args.topology,
        "transform_name": transform_name,
        "matrix": matrix_to_list(X),
        "sample_count": len(sample_ids),
        "motion_pair_count": len(motion_pairs),
        "sample_ids": sample_ids,
        "explanation": explanation,
        "diagnostics": diagnostics,
    }
    save_json(args.output, payload)

    print("Teaching solver complete.")
    print(f"  topology: {args.topology}")
    print(f"  samples: {len(sample_ids)}")
    print(f"  motion pairs: {len(motion_pairs)}")
    print(f"  skipped weak/degenerate pairs: {diagnostics['skipped_pair_count']}")
    print(f"  output saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
