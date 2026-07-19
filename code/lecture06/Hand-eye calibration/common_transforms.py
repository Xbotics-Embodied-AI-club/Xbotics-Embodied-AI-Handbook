"""Common SE(3) helpers for lecture 6 hand-eye calibration demos."""

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


def make_transform(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("rotation 必须是 3x3 矩阵。")
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
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


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
            "max_translation_mm": 0.0,
            "mean_rotation_deg": 0.0,
            "max_rotation_deg": 0.0,
        }

    trans_mm = np.array([item[0] * 1000.0 for item in errors], dtype=np.float64)
    rot_deg = np.array([item[1] for item in errors], dtype=np.float64)
    return {
        "count": int(len(errors)),
        "mean_translation_mm": float(trans_mm.mean()),
        "max_translation_mm": float(trans_mm.max()),
        "mean_rotation_deg": float(rot_deg.mean()),
        "max_rotation_deg": float(rot_deg.max()),
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
        records.append({"id": pose_id, "matrix_key": key, "matrix": T})

    return data, records


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
        raise ValueError("配对后的有效样本数不足 3，无法做手眼标定。")
    return ids, robot_mats, camera_mats


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
