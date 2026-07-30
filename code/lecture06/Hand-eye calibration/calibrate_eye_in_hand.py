"""Eye-in-Hand hand-eye calibration demo for lecture 6."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from common_transforms import (
    build_pairing_report,
    build_relative_motion_pairs,
    diagnose_motion_dataset,
    find_weak_motion_pairs,
    load_pose_records,
    pair_pose_records,
    save_json,
    save_transform_result,
    solve_hand_eye_park,
    summarize_motion_pairs,
    summarize_transform_constancy,
)


METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Eye-in-Hand hand-eye calibration.")
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
    parser.add_argument("--output", required=True, help="Path to the solved transform JSON.")
    parser.add_argument(
        "--method",
        default="tsai",
        choices=list(METHOD_MAP.keys()),
        help="cv2.calibrateHandEye method name.",
    )
    parser.add_argument(
        "--solver",
        default="opencv",
        choices=["opencv", "park_teaching"],
        help="opencv 为默认工程解；park_teaching 为显式展示 AX=XB 逻辑的教学解。",
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
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional path to residual summary JSON.",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional path to save a richer teaching report JSON.",
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
    return parser.parse_args()


def build_motion_pairs_eye_in_hand(sample_ids: Sequence[str], robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray]):
    return build_relative_motion_pairs(sample_ids, robot_mats, camera_mats)


def solve_with_opencv(robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray], method: str) -> np.ndarray:
    if len(robot_mats) < 3:
        raise ValueError("有效样本数量不足，无法求解。")

    gripper_R = [item[:3, :3] for item in robot_mats]
    gripper_t = [item[:3, 3].reshape(3, 1) for item in robot_mats]
    target_R = [item[:3, :3] for item in camera_mats]
    target_t = [item[:3, 3].reshape(3, 1) for item in camera_mats]

    method_flag = METHOD_MAP[method]
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        gripper_R,
        gripper_t,
        target_R,
        target_t,
        method=method_flag,
    )

    T_end_camera = np.eye(4, dtype=np.float64)
    T_end_camera[:3, :3] = R_cam2gripper
    T_end_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    return T_end_camera


def evaluate_eye_in_hand(sample_ids: Sequence[str], robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray], T_end_camera: np.ndarray):
    base_board = [robot @ T_end_camera @ camera for robot, camera in zip(robot_mats, camera_mats)]
    return summarize_transform_constancy(sample_ids, base_board)


def main() -> None:
    args = parse_args()

    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    pairing_report = build_pairing_report(robot_records, camera_records)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    motion_pairs = build_motion_pairs_eye_in_hand(sample_ids, robot_mats, camera_mats)
    weak_pairs = find_weak_motion_pairs(
        motion_pairs,
        min_rotation_deg=args.min_rotation_deg,
        min_translation_mm=args.min_translation_mm,
    )
    motion_summary = summarize_motion_pairs(motion_pairs)
    diagnostics = diagnose_motion_dataset(
        motion_summary,
        len(weak_pairs),
        len(sample_ids),
        min_rotation_deg=args.min_rotation_deg,
        min_translation_mm=args.min_translation_mm,
    )

    teaching_diagnostics = None
    if args.solver == "opencv":
        T_end_camera = solve_with_opencv(robot_mats, camera_mats, args.method)
        solver_name = f"opencv_calibrateHandEye:{args.method}"
    else:
        T_end_camera, teaching_diagnostics = solve_hand_eye_park(motion_pairs)
        solver_name = "park_teaching"

    validation = evaluate_eye_in_hand(sample_ids, robot_mats, camera_mats, T_end_camera)
    residual_summary = validation["summary"]

    metadata = {
        "robot_matrix_key": args.robot_matrix_key,
        "camera_matrix_key": args.camera_matrix_key,
        "kinematic_chain": "T_base_board = T_base_end · T_end_camera · T_cam_board",
        "pairing": pairing_report,
        "motion_summary": motion_summary,
        "weak_motion_pair_count": len(weak_pairs),
        "weak_motion_threshold": {
            "min_rotation_deg": args.min_rotation_deg,
            "min_translation_mm": args.min_translation_mm,
        },
        "diagnostics": diagnostics,
    }
    if weak_pairs:
        metadata["weak_motion_pairs"] = weak_pairs
    if teaching_diagnostics is not None:
        metadata["teaching_solver"] = teaching_diagnostics

    save_transform_result(
        args.output,
        transform_name="T_end_camera",
        topology="eye_in_hand",
        matrix=T_end_camera,
        unit="meter",
        solver=solver_name,
        sample_ids=sample_ids,
        motion_pair_count=len(motion_pairs),
        residual_summary=residual_summary,
        extra_metadata=metadata,
    )

    if args.summary_output:
        save_json(args.summary_output, residual_summary)

    if args.report_output:
        report = {
            "transform_name": "T_end_camera",
            "topology": "eye_in_hand",
            "solver": solver_name,
            "sample_ids": sample_ids,
            "pairing": pairing_report,
            "motion_summary": motion_summary,
            "weak_motion_pairs": weak_pairs,
            "diagnostics": diagnostics,
            "board_constancy": validation,
        }
        if teaching_diagnostics is not None:
            report["teaching_solver"] = teaching_diagnostics
        save_json(args.report_output, report)

    print("Eye-in-Hand calibration complete.")
    print("  chain: T_base_board = T_base_end · T_end_camera · T_cam_board")
    print(f"  samples: {len(sample_ids)}")
    print(f"  motion pairs: {len(motion_pairs)}")
    print(f"  weak motion pairs: {len(weak_pairs)}")
    if diagnostics:
        print("  diagnostics:")
        for item in diagnostics:
            print(f"    - {item}")
    print(f"  mean translation residual: {residual_summary['mean_translation_mm']:.3f} mm")
    print(f"  mean rotation residual: {residual_summary['mean_rotation_deg']:.4f} deg")
    print(f"  result saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
