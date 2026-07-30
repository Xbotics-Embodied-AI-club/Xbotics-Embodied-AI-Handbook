"""Shared SE(3) helpers for lecture 6 hand-eye calibration demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MATRIX_KEYS = (
    "matrix",
    "T_base_end",
    "T_cam_board",
    "T_end_board",
    "T_end_camera",
    "T_base_camera",
)


def _to_array(matrix: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} 必须是 4x4 齐次矩阵，当前形状为 {arr.shape}。")
    return arr


def validate_transform(T: np.ndarray, name: str = "transform") -> np.ndarray:
    arr = _to_array(T, name)
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} 的最后一行必须是 [0, 0, 0, 1]。")

    R = arr[:3, :3]
    should_be_identity = R.T @ R
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} 的旋转部分不是正交矩阵。")
    if np.linalg.det(R) <= 0.0:
        raise ValueError(f"{name} 的旋转部分行列式必须为正。")
    return arr


def validate_rotation(R: np.ndarray, name: str = "rotation") -> np.ndarray:
    arr = np.asarray(R, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} 必须是 3x3 矩阵，当前形状为 {arr.shape}。")
    should_be_identity = arr.T @ arr
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} 不是正交矩阵。")
    if np.linalg.det(arr) <= 0.0:
        raise ValueError(f"{name} 的行列式必须为正。")
    return arr


def make_transform(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    validate_rotation(R, "rotation")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return validate_transform(T, "make_transform() 输出")


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = validate_transform(T)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def compose(*transforms: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    for index, T in enumerate(transforms, start=1):
        result = result @ validate_transform(T, f"compose 输入第 {index} 个矩阵")
    return validate_transform(result, "compose() 输出")


def rotation_angle_deg(R: np.ndarray) -> float:
    R = validate_rotation(R)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def axis_angle_to_rotation(axis_angle: Sequence[float]) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = vec / theta
    K = _skew(axis)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return validate_rotation(R, "axis_angle_to_rotation() 输出")


def axis_angle_vector_from_rotation(R: np.ndarray) -> np.ndarray:
    R = validate_rotation(R)
    theta_deg = rotation_angle_deg(R)
    theta = float(np.radians(theta_deg))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if np.pi - theta < 1e-6:
        eigvals, eigvecs = np.linalg.eig(R)
        index = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, index])
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    vee = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis = vee / (2.0 * np.sin(theta))
    return axis * theta


def quaternion_to_rotation_matrix(quaternion: Sequence[float], *, order: str = "xyzw") -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError("order 只能是 'xyzw' 或 'wxyz'。")

    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError("四元数范数不能为 0。")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return validate_rotation(R, "quaternion_to_rotation_matrix() 输出")


def quaternion_xyz_to_transform(
    translation_xyz: Sequence[float],
    quaternion: Sequence[float],
    *,
    order: str = "xyzw",
) -> np.ndarray:
    return make_transform(quaternion_to_rotation_matrix(quaternion, order=order), translation_xyz)


def rpy_xyz_to_transform(
    translation_xyz: Sequence[float],
    rpy: Sequence[float],
    *,
    degrees: bool = False,
) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return make_transform(Rz @ Ry @ Rx, translation_xyz)


def rvec_tvec_to_transform(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    R = axis_angle_to_rotation(np.asarray(rvec, dtype=np.float64).reshape(3))
    return make_transform(R, tvec)


def pose_error(T_ref: np.ndarray, T: np.ndarray) -> Tuple[float, float]:
    T_ref = validate_transform(T_ref, "T_ref")
    T = validate_transform(T, "T")
    dt = float(np.linalg.norm(T_ref[:3, 3] - T[:3, 3]))
    dR = T_ref[:3, :3].T @ T[:3, :3]
    drot_deg = rotation_angle_deg(dR)
    return dt, drot_deg


def summarize_pose_errors(errors: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    errors = list(errors)
    if not errors:
        return {
            "count": 0,
            "mean_translation_mm": 0.0,
            "median_translation_mm": 0.0,
            "max_translation_mm": 0.0,
            "mean_rotation_deg": 0.0,
            "median_rotation_deg": 0.0,
            "max_rotation_deg": 0.0,
        }

    trans_mm = np.array([item[0] * 1000.0 for item in errors], dtype=np.float64)
    rot_deg = np.array([item[1] for item in errors], dtype=np.float64)
    return {
        "count": int(len(errors)),
        "mean_translation_mm": float(trans_mm.mean()),
        "median_translation_mm": float(np.median(trans_mm)),
        "max_translation_mm": float(trans_mm.max()),
        "mean_rotation_deg": float(rot_deg.mean()),
        "median_rotation_deg": float(np.median(rot_deg)),
        "max_rotation_deg": float(rot_deg.max()),
    }


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("average_transforms() 需要至少一个变换矩阵。")
    mats = [validate_transform(T, f"average_transforms 输入第 {i} 个矩阵") for i, T in enumerate(transforms, start=1)]
    mean_t = np.mean([T[:3, 3] for T in mats], axis=0)
    M = np.sum([T[:3, :3] for T in mats], axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return make_transform(R, mean_t)


def summarize_transform_constancy(sample_ids: Sequence[str], transforms: Sequence[np.ndarray]) -> dict:
    if len(sample_ids) != len(transforms):
        raise ValueError("sample_ids 与 transforms 数量不一致。")
    if not transforms:
        raise ValueError("至少需要一个变换矩阵做一致性分析。")

    reference = average_transforms(transforms)
    per_sample = []
    errors = []
    for sample_id, current in zip(sample_ids, transforms):
        dt, drot = pose_error(reference, current)
        errors.append((dt, drot))
        per_sample.append(
            {
                "id": sample_id,
                "translation_mm": float(dt * 1000.0),
                "rotation_deg": float(drot),
            }
        )

    worst_translation = max(per_sample, key=lambda item: item["translation_mm"])
    worst_rotation = max(per_sample, key=lambda item: item["rotation_deg"])
    return {
        "reference_transform": matrix_to_list(reference),
        "summary": summarize_pose_errors(errors),
        "per_sample": per_sample,
        "worst_translation_sample": worst_translation,
        "worst_rotation_sample": worst_rotation,
    }


def matrix_to_list(T: np.ndarray) -> List[List[float]]:
    return validate_transform(T).tolist()


def _detect_matrix_key(record: dict, matrix_key: Optional[str]) -> str:
    if matrix_key is not None:
        if matrix_key not in record:
            raise KeyError(f"记录中缺少矩阵字段 {matrix_key!r}。")
        return matrix_key

    for key in DEFAULT_MATRIX_KEYS:
        if key in record:
            return key
    raise KeyError(f"未找到矩阵字段。可选字段示例：{DEFAULT_MATRIX_KEYS}")


def load_pose_records(path: str, matrix_key: Optional[str] = None) -> Tuple[dict, List[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "poses" not in data or not isinstance(data["poses"], list):
        raise ValueError(f"{path} 必须包含 poses 列表。")

    records: List[dict] = []
    seen_ids = set()
    for index, raw in enumerate(data["poses"]):
        if not isinstance(raw, dict):
            raise ValueError(f"{path} 的第 {index} 条 pose 不是对象。")
        pose_id = raw.get("id")
        if not pose_id:
            raise ValueError(f"{path} 的第 {index} 条 pose 缺少 id。")
        if pose_id in seen_ids:
            raise ValueError(f"{path} 中存在重复 id: {pose_id}")
        seen_ids.add(pose_id)

        key = _detect_matrix_key(raw, matrix_key)
        T = validate_transform(raw[key], f"{path}:{pose_id}")
        metadata = {k: v for k, v in raw.items() if k != key}
        records.append({"id": pose_id, "matrix_key": key, "matrix": T, "metadata": metadata})

    return data, records


def load_transform_from_file(path: str, matrix_key: str = "matrix") -> Tuple[dict, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if matrix_key not in payload:
        raise KeyError(f"{path} 中没有找到 {matrix_key!r}。")
    return payload, validate_transform(payload[matrix_key], f"{path}:{matrix_key}")


def pair_pose_records(
    robot_records: Sequence[dict],
    camera_records: Sequence[dict],
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    camera_map = {item["id"]: item["matrix"] for item in camera_records}
    ids: List[str] = []
    robot_mats: List[np.ndarray] = []
    camera_mats: List[np.ndarray] = []

    for item in robot_records:
        pose_id = item["id"]
        if pose_id not in camera_map:
            continue
        ids.append(pose_id)
        robot_mats.append(item["matrix"])
        camera_mats.append(camera_map[pose_id])

    if len(ids) < 3:
        report = build_pairing_report(robot_records, camera_records)
        raise ValueError(
            "配对后的有效样本数不足 3，无法做手眼标定。"
            f" paired={report['paired_count']}, "
            f"missing_in_camera={report['missing_in_camera']}, "
            f"missing_in_robot={report['missing_in_robot']}。"
        )
    return ids, robot_mats, camera_mats


def build_pairing_report(robot_records: Sequence[dict], camera_records: Sequence[dict]) -> dict:
    """汇总机器人侧与相机侧样本配对情况。"""
    robot_ids = {item["id"] for item in robot_records}
    camera_ids = {item["id"] for item in camera_records}
    paired_ids = sorted(robot_ids & camera_ids)
    return {
        "robot_count": int(len(robot_records)),
        "camera_count": int(len(camera_records)),
        "paired_count": int(len(paired_ids)),
        "paired_ids": paired_ids,
        "missing_in_camera": sorted(robot_ids - camera_ids),
        "missing_in_robot": sorted(camera_ids - robot_ids),
    }


def build_relative_motion_pairs(
    sample_ids: Sequence[str],
    robot_mats: Sequence[np.ndarray],
    camera_mats: Sequence[np.ndarray],
) -> List[dict]:
    if not (len(sample_ids) == len(robot_mats) == len(camera_mats)):
        raise ValueError("sample_ids、robot_mats、camera_mats 数量必须一致。")

    pairs: List[dict] = []
    n = len(sample_ids)
    for i in range(n - 1):
        for j in range(i + 1, n):
            A = invert_transform(robot_mats[j]) @ robot_mats[i]
            B = camera_mats[j] @ invert_transform(camera_mats[i])
            pairs.append(
                {
                    "first_id": sample_ids[i],
                    "second_id": sample_ids[j],
                    "pair_id": f"{sample_ids[i]}->{sample_ids[j]}",
                    "A": validate_transform(A, f"A[{sample_ids[i]}->{sample_ids[j]}]"),
                    "B": validate_transform(B, f"B[{sample_ids[i]}->{sample_ids[j]}]"),
                }
            )
    return pairs


def summarize_motion_pairs(motion_pairs: Sequence[dict]) -> dict:
    if not motion_pairs:
        return {
            "pair_count": 0,
            "robot_rotation_deg": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "robot_translation_mm": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "camera_rotation_deg": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "camera_translation_mm": {"min": 0.0, "mean": 0.0, "max": 0.0},
        }

    robot_rot = np.array([rotation_angle_deg(item["A"][:3, :3]) for item in motion_pairs], dtype=np.float64)
    robot_trans = np.array([np.linalg.norm(item["A"][:3, 3]) * 1000.0 for item in motion_pairs], dtype=np.float64)
    camera_rot = np.array([rotation_angle_deg(item["B"][:3, :3]) for item in motion_pairs], dtype=np.float64)
    camera_trans = np.array([np.linalg.norm(item["B"][:3, 3]) * 1000.0 for item in motion_pairs], dtype=np.float64)

    def _summary(values: np.ndarray) -> dict:
        return {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    return {
        "pair_count": int(len(motion_pairs)),
        "robot_rotation_deg": _summary(robot_rot),
        "robot_translation_mm": _summary(robot_trans),
        "camera_rotation_deg": _summary(camera_rot),
        "camera_translation_mm": _summary(camera_trans),
    }


def find_weak_motion_pairs(
    motion_pairs: Sequence[dict],
    *,
    min_rotation_deg: float = 5.0,
    min_translation_mm: float = 5.0,
) -> List[dict]:
    flagged = []
    for item in motion_pairs:
        robot_rotation = rotation_angle_deg(item["A"][:3, :3])
        robot_translation_mm = float(np.linalg.norm(item["A"][:3, 3]) * 1000.0)
        camera_rotation = rotation_angle_deg(item["B"][:3, :3])
        camera_translation_mm = float(np.linalg.norm(item["B"][:3, 3]) * 1000.0)
        if (
            robot_rotation < min_rotation_deg
            or robot_translation_mm < min_translation_mm
            or camera_rotation < min_rotation_deg
            or camera_translation_mm < min_translation_mm
        ):
            flagged.append(
                {
                    "pair_id": item["pair_id"],
                    "robot_rotation_deg": robot_rotation,
                    "robot_translation_mm": robot_translation_mm,
                    "camera_rotation_deg": camera_rotation,
                    "camera_translation_mm": camera_translation_mm,
                }
            )
    return flagged


def diagnose_motion_dataset(
    motion_summary: dict,
    weak_motion_pair_count: int,
    sample_count: int,
    *,
    min_rotation_deg: float = 5.0,
    min_translation_mm: float = 5.0,
) -> List[str]:
    """根据样本数、相对运动范围和弱运动比例生成可读诊断建议。"""
    pair_count = int(motion_summary.get("pair_count", 0))
    warnings: List[str] = []

    if sample_count < 8:
        warnings.append("样本数偏少。教学演示可以运行，真实标定建议采集 15-30 组。")
    if pair_count < 10:
        warnings.append("相对运动对数量偏少，求解结果对单个异常样本会更敏感。")

    if pair_count > 0 and weak_motion_pair_count / pair_count > 0.5:
        warnings.append("超过一半相对运动对低于阈值，建议增加姿态变化更明显的样本。")

    robot_rot_max = motion_summary["robot_rotation_deg"]["max"]
    camera_rot_max = motion_summary["camera_rotation_deg"]["max"]
    robot_trans_max = motion_summary["robot_translation_mm"]["max"]
    camera_trans_max = motion_summary["camera_translation_mm"]["max"]

    if robot_rot_max < 2.0 * min_rotation_deg or camera_rot_max < 2.0 * min_rotation_deg:
        warnings.append("旋转激励不足：末端和标定板观测需要覆盖更明显的 roll/pitch/yaw 变化。")
    if robot_trans_max < 2.0 * min_translation_mm or camera_trans_max < 2.0 * min_translation_mm:
        warnings.append("平移激励不足：采样位置过于集中，建议覆盖更大的工作空间。")

    return warnings


def solve_hand_eye_park(motion_pairs: Sequence[dict]) -> Tuple[np.ndarray, dict]:
    if len(motion_pairs) < 2:
        raise ValueError("至少需要 2 个相对运动对才能进行教学版求解。")

    usable_pairs = []
    skipped_pairs = []
    alpha_list = []
    beta_list = []
    for item in motion_pairs:
        alpha = axis_angle_vector_from_rotation(item["A"][:3, :3])
        beta = axis_angle_vector_from_rotation(item["B"][:3, :3])
        if np.linalg.norm(alpha) < 1e-8 or np.linalg.norm(beta) < 1e-8:
            skipped_pairs.append(item["pair_id"])
            continue
        usable_pairs.append(item)
        alpha_list.append(alpha)
        beta_list.append(beta)

    if len(usable_pairs) < 2:
        raise ValueError("可用于求解旋转的有效运动对不足。请增加姿态变化更明显的样本。")

    source = np.stack(beta_list, axis=0)
    target = np.stack(alpha_list, axis=0)
    H = source.T @ target
    U, singular_values, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    R = validate_rotation(R, "solve_hand_eye_park rotation")

    lhs = []
    rhs = []
    for item in usable_pairs:
        A = item["A"]
        B = item["B"]
        lhs.append(A[:3, :3] - np.eye(3))
        rhs.append(R @ B[:3, 3] - A[:3, 3])
    lhs_mat = np.vstack(lhs)
    rhs_vec = np.concatenate(rhs)
    t, residuals, rank, singular_values_lstsq = np.linalg.lstsq(lhs_mat, rhs_vec, rcond=None)
    X = make_transform(R, t)

    motion_residuals = []
    pose_errors = []
    for item in usable_pairs:
        lhs_T = item["A"] @ X
        rhs_T = X @ item["B"]
        dt, drot = pose_error(lhs_T, rhs_T)
        pose_errors.append((dt, drot))
        motion_residuals.append(
            {
                "pair_id": item["pair_id"],
                "translation_mm": float(dt * 1000.0),
                "rotation_deg": float(drot),
            }
        )

    diagnostics = {
        "solver": "park_teaching",
        "used_pair_count": int(len(usable_pairs)),
        "skipped_pair_count": int(len(skipped_pairs)),
        "skipped_pair_ids": skipped_pairs,
        "rotation_singular_values": [float(v) for v in singular_values],
        "translation_lstsq_rank": int(rank),
        "translation_lstsq_singular_values": [float(v) for v in singular_values_lstsq],
        "translation_lstsq_residual": float(residuals[0]) if residuals.size else 0.0,
        "motion_equation_residual_summary": summarize_pose_errors(pose_errors),
        "motion_equation_residuals": motion_residuals,
    }
    return X, diagnostics


def save_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_transform_result(
    path: str,
    *,
    transform_name: str,
    topology: str,
    matrix: np.ndarray,
    unit: str,
    solver: str,
    sample_ids: Sequence[str],
    motion_pair_count: int,
    residual_summary: dict,
    extra_metadata: Optional[dict] = None,
) -> None:
    payload = {
        "transform_name": transform_name,
        "topology": topology,
        "unit": unit,
        "solver": solver,
        "sample_count": int(len(sample_ids)),
        "motion_pair_count": int(motion_pair_count),
        "sample_ids": list(sample_ids),
        "matrix": matrix_to_list(matrix),
        "residual_summary": residual_summary,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    save_json(path, payload)
