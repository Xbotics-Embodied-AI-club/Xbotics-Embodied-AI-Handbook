"""One-command hand-eye calibration demo pipeline for lecture 6."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common_transforms import load_transform_from_file, pose_error, save_json, validate_transform


TRANSFORM_NAME = {
    "eye_in_hand": "T_end_camera",
    "eye_to_hand": "T_base_camera",
}

CALIBRATION_SCRIPT = {
    "eye_in_hand": "calibrate_eye_in_hand.py",
    "eye_to_hand": "calibrate_eye_to_hand.py",
}

CHAIN_TEXT = {
    "eye_in_hand": "T_base_board = T_base_end * T_end_camera * T_cam_board",
    "eye_to_hand": "T_base_end * T_end_board = T_base_camera * T_cam_board",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a complete synthetic hand-eye calibration demo.")
    parser.add_argument("--topology", required=True, choices=["eye_in_hand", "eye_to_hand"])
    parser.add_argument(
        "--demo-dir",
        default="",
        help="Output directory. Defaults to data/hand_eye_demo/<topology>.",
    )
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--noise-translation-mm", type=float, default=0.5)
    parser.add_argument("--noise-rotation-deg", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="tsai", choices=["tsai", "park", "horaud", "andreff", "daniilidis"])
    parser.add_argument(
        "--solver",
        default="opencv",
        choices=["opencv", "park_teaching"],
        help="Use OpenCV engineering solver or the teaching Park solver.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing robot_poses.json and camera_poses.json in --demo-dir.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_with_ground_truth(result_path: Path, truth_path: Path) -> dict:
    _, result_T = load_transform_from_file(str(result_path), matrix_key="matrix")
    truth = load_json(truth_path)
    truth_T = validate_transform(np.asarray(truth["matrix"], dtype=np.float64), "ground_truth.matrix")
    translation_m, rotation_deg = pose_error(truth_T, result_T)
    return {
        "transform_name": truth["transform_name"],
        "translation_error_mm": float(translation_m * 1000.0),
        "rotation_error_deg": float(rotation_deg),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    root = Path(__file__).resolve().parent
    demo_dir = Path(args.demo_dir) if args.demo_dir else Path("data") / "hand_eye_demo" / args.topology
    demo_dir = demo_dir if demo_dir.is_absolute() else root / demo_dir
    demo_dir.mkdir(parents=True, exist_ok=True)

    robot_poses = demo_dir / "robot_poses.json"
    camera_poses = demo_dir / "camera_poses.json"
    ground_truth = demo_dir / "ground_truth.json"
    inspect_report = demo_dir / "inspect_report.json"
    calibration_result = demo_dir / f"{args.topology}.json"
    calibration_report = demo_dir / f"{args.topology}_calibration_report.json"
    validation_report = demo_dir / "validation_report.json"
    pipeline_summary = demo_dir / "pipeline_summary.json"

    print("=" * 64)
    print("  Hand-eye calibration demo pipeline")
    print("=" * 64)
    print(f"  topology: {args.topology}")
    print(f"  chain: {CHAIN_TEXT[args.topology]}")
    print(f"  demo dir: {demo_dir}")

    if not args.skip_generate:
        print("\n[1/4] Generate synthetic paired robot/camera samples")
        run_command(
            [
                sys.executable,
                "generate_synthetic_hand_eye_data.py",
                "--topology",
                args.topology,
                "--output-dir",
                str(demo_dir),
                "--sample-count",
                str(args.sample_count),
                "--noise-translation-mm",
                str(args.noise_translation_mm),
                "--noise-rotation-deg",
                str(args.noise_rotation_deg),
                "--seed",
                str(args.seed),
            ],
            cwd=root,
        )
    else:
        print("\n[1/4] Reuse existing paired robot/camera samples")
        missing = [path for path in (robot_poses, camera_poses) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"--skip-generate requires existing files: {missing}")
        print(f"  robot poses: {robot_poses}")
        print(f"  camera poses: {camera_poses}")

    print("\n[2/4] Inspect sample pairing and motion excitation")
    run_command(
        [
            sys.executable,
            "inspect_hand_eye_dataset.py",
            "--topology",
            args.topology,
            "--robot-poses",
            str(robot_poses),
            "--camera-poses",
            str(camera_poses),
            "--report",
            str(inspect_report),
        ],
        cwd=root,
    )

    print("\n[3/4] Solve fixed hand-eye transform")
    run_command(
        [
            sys.executable,
            CALIBRATION_SCRIPT[args.topology],
            "--robot-poses",
            str(robot_poses),
            "--camera-poses",
            str(camera_poses),
            "--output",
            str(calibration_result),
            "--report-output",
            str(calibration_report),
            "--method",
            args.method,
            "--solver",
            args.solver,
        ],
        cwd=root,
    )

    print("\n[4/4] Validate closed-loop transform constancy")
    run_command(
        [
            sys.executable,
            "validate_hand_eye.py",
            "--transform",
            str(calibration_result),
            "--robot-poses",
            str(robot_poses),
            "--camera-poses",
            str(camera_poses),
            "--report",
            str(validation_report),
        ],
        cwd=root,
    )

    result = load_json(calibration_result)
    validation = load_json(validation_report)
    inspect = load_json(inspect_report)
    truth_error = compare_with_ground_truth(calibration_result, ground_truth) if ground_truth.exists() else None

    summary = {
        "topology": args.topology,
        "transform_name": TRANSFORM_NAME[args.topology],
        "chain": CHAIN_TEXT[args.topology],
        "demo_dir": str(demo_dir),
        "sample_count": result["sample_count"],
        "motion_pair_count": result["motion_pair_count"],
        "weak_motion_pair_count": inspect["weak_motion_pair_count"],
        "calibration_residual_summary": result["residual_summary"],
        "validation_residual_summary": validation["residual_summary"],
        "ground_truth_error": truth_error,
        "outputs": {
            "robot_poses": str(robot_poses),
            "camera_poses": str(camera_poses),
            "ground_truth": str(ground_truth),
            "inspect_report": str(inspect_report),
            "calibration_result": str(calibration_result),
            "calibration_report": str(calibration_report),
            "validation_report": str(validation_report),
            "pipeline_summary": str(pipeline_summary),
        },
    }
    save_json(str(pipeline_summary), summary)

    cal = result["residual_summary"]
    val = validation["residual_summary"]
    print("\n" + "=" * 64)
    print("  Pipeline complete")
    print("=" * 64)
    print(f"  solved transform: {TRANSFORM_NAME[args.topology]}")
    print(f"  samples: {result['sample_count']}")
    print(f"  motion pairs: {result['motion_pair_count']}")
    print(f"  weak motion pairs: {inspect['weak_motion_pair_count']}")
    print(f"  calibration mean residual: {cal['mean_translation_mm']:.3f} mm, {cal['mean_rotation_deg']:.4f} deg")
    print(f"  validation mean residual:  {val['mean_translation_mm']:.3f} mm, {val['mean_rotation_deg']:.4f} deg")
    if truth_error:
        print(
            "  ground-truth error:      "
            f"{truth_error['translation_error_mm']:.3f} mm, {truth_error['rotation_error_deg']:.4f} deg"
        )
    print(f"  result: {calibration_result}")
    print(f"  summary: {pipeline_summary}")


if __name__ == "__main__":
    main()
