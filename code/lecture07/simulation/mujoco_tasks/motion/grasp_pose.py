"""GraspNet-compatible 6D grasp pose (camera / world frame, meters)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRASP_ARRAY_LEN = 17


@dataclass(frozen=True, slots=True)
class GraspPose:
    """Single 6D parallel-jaw grasp in the GraspNet frame convention.

    Axes of ``rotation_matrix`` (columns), identical to the wrist-roll body frame:
    - X, Z: grasp plane (wrist-roll X-Z plane); jaw width is along X
    - Y: wrist-roll Y axis (plane normal)
    - Z: wrist-roll rotation axis; approach toward the object is ``-Z``
  """

    score: float
    width: float
    height: float
    depth: float
    rotation_matrix: np.ndarray
    translation: np.ndarray
    object_id: int = -1

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_matrix, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_array(cls, grasp_array: np.ndarray) -> GraspPose:
        values = np.asarray(grasp_array, dtype=np.float64).reshape(-1)
        if values.size != GRASP_ARRAY_LEN:
            raise ValueError(f"expected grasp array length {GRASP_ARRAY_LEN}, got {values.size}")
        return cls(
            score=float(values[0]),
            width=float(values[1]),
            height=float(values[2]),
            depth=float(values[3]),
            rotation_matrix=values[4:13].reshape(3, 3),
            translation=values[13:16],
            object_id=int(values[16]),
        )

    def to_array(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array([self.score, self.width, self.height, self.depth], dtype=np.float64),
                self.rotation_matrix.reshape(-1),
                self.translation,
                np.array([self.object_id], dtype=np.float64),
            ]
        )

    def score_rgba(self) -> tuple[float, float, float, float]:
        """Match GraspNet ``plot_gripper_pro_max`` score coloring (red = high)."""

        score = float(np.clip(self.score, 0.0, 1.0))
        return score, 0.0, 1.0 - score, 0.85

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """SO-101 EEF pose that realizes this GraspNet grasp."""

        from .grasp_to_gripper import grasp_to_eef_pose

        return grasp_to_eef_pose(self)

    def gripper_qpos(
        self,
        *,
        gripper_range: tuple[float, float] | None = None,
    ) -> float:
        """SO-101 gripper joint angle for this grasp width."""

        from .grasp_to_gripper import grasp_width_to_gripper_qpos

        return grasp_width_to_gripper_qpos(self.width, gripper_range=gripper_range)
