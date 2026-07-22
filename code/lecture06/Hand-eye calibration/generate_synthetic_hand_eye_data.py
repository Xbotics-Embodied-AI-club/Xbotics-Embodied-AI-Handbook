"""Generate synthetic hand-eye calibration datasets for teaching and regression tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_transforms import axis_angle_to_rotation, invert_transform, make_transform, matrix_to_list, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for hand-eye calibration demos.")
    parser.add_argument("--topology", required=True, choices=["eye_in_hand", "eye_to_hand"])
    parser.add_argument("--output-dir", required=True, help="Directory to write robot/camera JSON files.")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of synthetic samples.")
    parser.add_argument("--noise-translation-mm", type=float, default=0.0, help="Gaussian noise on translation.")
    parser.add_argument("--noise-rotation-deg", type=float, default=0.0, help="Gaussian noise on rotation angle.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def random_transform(rng: np.random.Generator, translation_scale: float, rotation_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-rotation_deg, rotation_deg))
    rotation = axis_angle_to_rotation(axis * angle)
    translation = rng.uniform(-translation_scale, translation_scale, size=3)
    return make_transform(rotation, translation)


def noisy_transform(T: np.ndarray, rng: np.random.Generator, translation_noise_m: float, rotation_noise_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    angle = np.radians(rng.normal(scale=rotation_noise_deg))
    noise_R = axis_angle_to_rotation(axis * angle)
    noise_t = rng.normal(scale=translation_noise_m, size=3)
    noise = make_transform(noise_R, noise_t)
    return T @ noise


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_count = max(args.sample_count, 4)
    translation_noise_m = args.noise_translation_mm / 1000.0

    T_end_camera_gt = random_transform(rng, translation_scale=0.15, rotation_deg=35.0)
    T_base_camera_gt = random_transform(rng, translation_scale=0.40, rotation_deg=35.0)
    T_base_board_gt = random_transform(rng, translation_scale=0.35, rotation_deg=25.0)
    T_end_board_gt = random_transform(rng, translation_scale=0.12, rotation_deg=25.0)

    robot_poses = []
    camera_poses = []
    for index in range(sample_count):
        sample_id = f"sample_{index + 1:03d}"
        T_base_end = random_transform(rng, translation_scale=0.35, rotation_deg=55.0)

        if args.topology == "eye_in_hand":
            T_cam_board = invert_transform(T_end_camera_gt) @ invert_transform(T_base_end) @ T_base_board_gt
            truth_transform_name = "T_end_camera"
            truth_transform = T_end_camera_gt
        else:
            T_cam_board = invert_transform(T_base_camera_gt) @ T_base_end @ T_end_board_gt
            truth_transform_name = "T_base_camera"
            truth_transform = T_base_camera_gt

        T_cam_board = noisy_transform(
            T_cam_board,
            rng,
            translation_noise_m=translation_noise_m,
            rotation_noise_deg=args.noise_rotation_deg,
        )

        robot_poses.append({"id": sample_id, "T_base_end": matrix_to_list(T_base_end)})
        camera_poses.append({"id": sample_id, "T_cam_board": matrix_to_list(T_cam_board)})

    save_json(str(out_dir / "robot_poses.json"), {"poses": robot_poses})
    save_json(str(out_dir / "camera_poses.json"), {"poses": camera_poses})
    save_json(
        str(out_dir / "ground_truth.json"),
        {
            "topology": args.topology,
            "transform_name": truth_transform_name,
            "matrix": matrix_to_list(truth_transform),
            "noise_translation_mm": args.noise_translation_mm,
            "noise_rotation_deg": args.noise_rotation_deg,
            "sample_count": sample_count,
        },
    )

    print("Synthetic dataset generated.")
    print(f"  topology: {args.topology}")
    print(f"  samples: {sample_count}")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
