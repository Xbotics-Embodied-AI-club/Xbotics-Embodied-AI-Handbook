"""Estimate ArUco board poses and export camera_poses.json for hand-eye calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from common_transforms import matrix_to_list, rvec_tvec_to_transform, save_json


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
ARUCO_DICT_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a single ArUco marker and export T_cam_board poses for hand-eye calibration."
    )
    parser.add_argument(
        "--image",
        default="",
        help="Optional single image path. If omitted, use --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Directory containing RGB images to process.",
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="JSON file containing fx/fy/cx/cy and optional dist_coeffs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output camera_poses.json path.",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional detailed per-image report JSON path.",
    )
    parser.add_argument(
        "--marker-size-m",
        type=float,
        required=True,
        help="Marker side length in meters.",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="Target marker id to keep.",
    )
    parser.add_argument(
        "--aruco-dict",
        default="4x4_50",
        choices=sorted(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary name.",
    )
    parser.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=2.0,
        help="Reject detections whose mean reprojection error exceeds this threshold.",
    )
    parser.add_argument(
        "--dist-coeffs",
        default="",
        help="Optional comma-separated distortion coefficients overriding intrinsics.json.",
    )
    return parser.parse_args()


def load_intrinsics(path: str, dist_override: str) -> Tuple[np.ndarray, np.ndarray]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ["fx", "fy", "cx", "cy"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} 缺少内参字段: {missing}")

    K = np.array(
        [
            [float(payload["fx"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fy"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if dist_override:
        dist = np.array([float(item) for item in dist_override.split(",") if item.strip()], dtype=np.float64)
    elif "dist_coeffs" in payload:
        dist = np.asarray(payload["dist_coeffs"], dtype=np.float64).reshape(-1)
    elif "dist" in payload:
        dist = np.asarray(payload["dist"], dtype=np.float64).reshape(-1)
    else:
        dist = np.zeros(5, dtype=np.float64)
    return K, dist


def list_images(single_image: str, input_dir: str) -> List[Path]:
    if single_image and input_dir:
        raise ValueError("--image 和 --input-dir 只能二选一。")
    if single_image:
        return [Path(single_image)]
    if not input_dir:
        raise ValueError("必须提供 --image 或 --input-dir。")

    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到输入目录: {directory}")

    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"{directory} 中未找到支持的图像文件。")
    return images


def aruco_object_points(marker_size_m: float) -> np.ndarray:
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    reproj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(reproj - image_points, axis=1)))


def detect_pose(
    image_path: Path,
    detector: cv2.aruco.ArucoDetector,
    target_marker_id: int,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    max_reproj_error_px: float,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "id": image_path.stem,
            "image_path": str(image_path),
            "accepted": False,
            "reason": "image_read_failed",
        }

    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or len(ids) == 0:
        return {
            "id": image_path.stem,
            "image_path": str(image_path),
            "accepted": False,
            "reason": "marker_not_found",
        }

    ids = ids.reshape(-1)
    target_indices = np.where(ids == target_marker_id)[0]
    if len(target_indices) == 0:
        return {
            "id": image_path.stem,
            "image_path": str(image_path),
            "accepted": False,
            "reason": "target_marker_id_not_found",
            "detected_marker_ids": ids.tolist(),
        }

    index = int(target_indices[0])
    image_points = corners[index].reshape(4, 2).astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return {
            "id": image_path.stem,
            "image_path": str(image_path),
            "accepted": False,
            "reason": "solvepnp_failed",
            "detected_marker_ids": ids.tolist(),
        }

    reproj_error = mean_reprojection_error(object_points, image_points, rvec, tvec, K, dist)
    T_cam_board = rvec_tvec_to_transform(rvec.reshape(3), tvec.reshape(3))
    accepted = reproj_error <= max_reproj_error_px
    return {
        "id": image_path.stem,
        "image_path": str(image_path),
        "accepted": accepted,
        "reason": "ok" if accepted else "reprojection_error_too_large",
        "marker_id": int(target_marker_id),
        "detected_marker_ids": ids.tolist(),
        "reprojection_error_px": reproj_error,
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "T_cam_board": matrix_to_list(T_cam_board),
    }


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.image, args.input_dir)
    K, dist = load_intrinsics(args.intrinsics, args.dist_coeffs)
    object_points = aruco_object_points(args.marker_size_m)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[args.aruco_dict])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    detections = [
        detect_pose(
            image_path=path,
            detector=detector,
            target_marker_id=args.marker_id,
            object_points=object_points,
            K=K,
            dist=dist,
            max_reproj_error_px=args.max_reproj_error_px,
        )
        for path in image_paths
    ]

    accepted = [
        {
            "id": item["id"],
            "T_cam_board": item["T_cam_board"],
            "marker_id": item["marker_id"],
            "marker_size_m": args.marker_size_m,
            "reprojection_error_px": item["reprojection_error_px"],
            "image_path": item["image_path"],
        }
        for item in detections
        if item["accepted"]
    ]

    camera_poses_payload = {
        "poses": accepted,
        "metadata": {
            "board_type": "aruco_single_marker",
            "aruco_dict": args.aruco_dict,
            "marker_id": args.marker_id,
            "marker_size_m": args.marker_size_m,
            "max_reproj_error_px": args.max_reproj_error_px,
        },
    }
    save_json(args.output, camera_poses_payload)

    if args.report_output:
        report = {
            "summary": {
                "image_count": len(image_paths),
                "accepted_count": len(accepted),
                "rejected_count": len(image_paths) - len(accepted),
            },
            "metadata": camera_poses_payload["metadata"],
            "intrinsics": {
                "K": K.tolist(),
                "dist_coeffs": dist.tolist(),
            },
            "detections": detections,
        }
        save_json(args.report_output, report)

    print("ArUco pose extraction complete.")
    print(f"  images: {len(image_paths)}")
    print(f"  accepted poses: {len(accepted)}")
    print(f"  output saved to: {Path(args.output)}")
    if args.report_output:
        print(f"  detailed report: {Path(args.report_output)}")


if __name__ == "__main__":
    main()
