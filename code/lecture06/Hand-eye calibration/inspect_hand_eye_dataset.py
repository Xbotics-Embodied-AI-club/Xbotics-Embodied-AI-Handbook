"""Inspect hand-eye calibration datasets before solving."""

from __future__ import annotations

import argparse
from pathlib import Path

from common_transforms import (
    build_relative_motion_pairs,
    find_weak_motion_pairs,
    load_pose_records,
    pair_pose_records,
    save_json,
    summarize_motion_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect paired robot/camera samples for hand-eye calibration.")
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
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
    parser.add_argument(
        "--min-rotation-deg",
        type=float,
        default=5.0,
        help="Threshold for flagging weak motion pairs.",
    )
    parser.add_argument(
        "--min-translation-mm",
        type=float,
        default=5.0,
        help="Threshold for flagging weak motion pairs.",
    )
    parser.add_argument("--report", default="", help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    motion_pairs = build_relative_motion_pairs(sample_ids, robot_mats, camera_mats)
    motion_summary = summarize_motion_pairs(motion_pairs)
    weak_pairs = find_weak_motion_pairs(
        motion_pairs,
        min_rotation_deg=args.min_rotation_deg,
        min_translation_mm=args.min_translation_mm,
    )

    robot_ids = {item["id"] for item in robot_records}
    camera_ids = {item["id"] for item in camera_records}
    report = {
        "sample_count": len(sample_ids),
        "paired_ids": sample_ids,
        "missing_in_camera": sorted(robot_ids - camera_ids),
        "missing_in_robot": sorted(camera_ids - robot_ids),
        "motion_summary": motion_summary,
        "weak_motion_threshold": {
            "min_rotation_deg": args.min_rotation_deg,
            "min_translation_mm": args.min_translation_mm,
        },
        "weak_motion_pair_count": len(weak_pairs),
        "weak_motion_pairs": weak_pairs,
    }

    if args.report:
        save_json(args.report, report)

    print("Dataset inspection complete.")
    print(f"  paired samples: {len(sample_ids)}")
    print(f"  motion pairs: {motion_summary['pair_count']}")
    print(f"  weak motion pairs: {len(weak_pairs)}")
    print(
        f"  robot rotation range: {motion_summary['robot_rotation_deg']['min']:.2f} ~ "
        f"{motion_summary['robot_rotation_deg']['max']:.2f} deg"
    )
    print(
        f"  robot translation range: {motion_summary['robot_translation_mm']['min']:.2f} ~ "
        f"{motion_summary['robot_translation_mm']['max']:.2f} mm"
    )
    if args.report:
        print(f"  report saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
