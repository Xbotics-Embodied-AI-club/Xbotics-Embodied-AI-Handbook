"""Eye-in-Hand hand-eye calibration demo for lecture 6."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from common_transforms import (
    invert_transform,
    load_pose_records,
    pair_pose_records,
    pose_error,
    save_transform_result,
    summarize_pose_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Eye-in-Hand hand-eye calibration.")
    parser.add_argument("--robot-poses", required=True, help="JSON file with T_base_end samples.")
    parser.add_argument("--camera-poses", required=True, help="JSON file with T_cam_board samples.")
    parser.add_argument("--output", required=True, help="Path to the solved transform JSON.")
    parser.add_argument(
        "--method",
        default="tsai",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        help="cv2.calibrateHandEye method name.")
    parser.add_argument(
        "--robot-matrix-key",
        default="T_base_end",
        help="Matrix key inside robot pose records.")
    parser.add_argument(
        "--camera-matrix-key",
        default="T_cam_board",
        help="Matrix key inside camera pose records.")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional path to residual summary JSON.")
    return parser.parse_args()


METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _split_rt(mats: Sequence[np.ndarray]):
    rotations = [m[:3, :3].astype(np.float64) for m in mats]
    translations = [m[:3, 3].astype(np.float64).reshape(3, 1) for m in mats]
    return rotations, translations


def build_motion_pairs_eye_in_hand(robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray]):
    a_rot, a_trans = [], []
    b_rot, b_trans = [], []
    n = len(robot_mats)
    for i in range(n - 1):
        for j in range(i + 1, n):
            A = invert_transform(robot_mats[j]) @ robot_mats[i]
            B = camera_mats[j] @ invert_transform(camera_mats[i])
            a_rot.append(A[:3, :3])
            a_trans.append(A[:3, 3].reshape(3, 1))
            b_rot.append(B[:3, :3])
            b_trans.append(B[:3, 3].reshape(3, 1))
    return a_rot, a_trans, b_rot, b_trans


def solve_with_opencv(robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray], method: str) -> np.ndarray:
    a_rot, a_trans, b_rot, b_trans = build_motion_pairs_eye_in_hand(robot_mats, camera_mats)
    if len(a_rot) < 3:
        raise ValueError("有效相对运动数量不足，无法求解。")

    method_flag = METHOD_MAP[method]
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        a_rot,
        a_trans,
        b_rot,
        b_trans,
        method=method_flag,
    )

    T_end_camera = np.eye(4, dtype=np.float64)
    T_end_camera[:3, :3] = R_cam2gripper
    T_end_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    return T_end_camera


def evaluate_eye_in_hand(robot_mats: Sequence[np.ndarray], camera_mats: Sequence[np.ndarray], T_end_camera: np.ndarray):
    base_board = [robot @ T_end_camera @ camera for robot, camera in zip(robot_mats, camera_mats)]
    ref = base_board[0]
    errors = [pose_error(ref, current) for current in base_board[1:]]
    return summarize_pose_errors(errors)


def main() -> None:
    args = parse_args()

    _, robot_records = load_pose_records(args.robot_poses, matrix_key=args.robot_matrix_key)
    _, camera_records = load_pose_records(args.camera_poses, matrix_key=args.camera_matrix_key)
    sample_ids, robot_mats, camera_mats = pair_pose_records(robot_records, camera_records)

    T_end_camera = solve_with_opencv(robot_mats, camera_mats, args.method)
    residual_summary = evaluate_eye_in_hand(robot_mats, camera_mats, T_end_camera)

    save_transform_result(
        args.output,
        transform_name="T_end_camera",
        topology="eye_in_hand",
        matrix=T_end_camera,
        unit="meter",
        solver=f"opencv_calibrateHandEye:{args.method}",
        sample_ids=sample_ids,
        motion_pair_count=len(robot_mats) * (len(robot_mats) - 1) // 2,
        residual_summary=residual_summary,
        extra_metadata={
            "robot_matrix_key": args.robot_matrix_key,
            "camera_matrix_key": args.camera_matrix_key,
        },
    )

    if args.summary_output:
        from common_transforms import save_json

        save_json(args.summary_output, residual_summary)

    print("Eye-in-Hand calibration complete.")
    print(f"  samples: {len(sample_ids)}")
    print(f"  mean translation residual: {residual_summary['mean_translation_mm']:.3f} mm")
    print(f"  mean rotation residual: {residual_summary['mean_rotation_deg']:.4f} deg")
    print(f"  result saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
