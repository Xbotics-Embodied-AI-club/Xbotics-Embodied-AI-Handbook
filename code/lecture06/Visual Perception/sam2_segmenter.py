# sam2_segmenter.py
# 功能：SAM2 分割模型的统一包装层
# - Sam2SegmentationResult：分割结果数据类（掩膜 + 置信度 + 提示类型）
# - Sam2Segmenter：提供 point/box 提示的 segment() 统一接口
# - 当前实现含两种回退掩膜逻辑：
#   - _point_seed_mask()：以点击位置为圆心的圆形掩膜
#   - _box_seed_mask()：框选区域的矩形掩膜
# - 预留了真实 SAM2 + torch 后端的扩展接口
# - 被 interactive_depth_pipeline.py 调用
#
# 依赖：numpy, 可选 torch + sam2

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class Sam2SegmentationResult:
    mask: np.ndarray
    score: Optional[float]
    prompt_type: str


class Sam2Segmenter:
    def __init__(self, checkpoint_path=None, config_path=None, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.backend = None
        self.predictor = None
        self._try_initialize_real_backend()

    def _try_initialize_real_backend(self):
        try:
            import torch  # noqa: F401
            import sam2  # noqa: F401
        except ModuleNotFoundError:
            self.backend = "fallback"
            return

        # 这里保留为可扩展接口：不同 SAM2 包的 predictor API 差异较大。
        # 当前仓库先提供统一包装层和回退逻辑，待用户安装并确认具体 SAM2 包后再补齐真实调用。
        self.backend = "stub_sam2"

    @staticmethod
    def _normalize_box(box_xyxy, image_shape):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _point_seed_mask(image_rgb, point_xy):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        x, y = int(point_xy[0]), int(point_xy[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        radius = max(18, min(h, w) // 14)
        yy, xx = np.ogrid[:h, :w]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        mask[circle] = True
        return mask

    @staticmethod
    def _box_seed_mask(image_rgb, box_xyxy):
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = Sam2Segmenter._normalize_box(box_xyxy, image_rgb.shape)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask

    def predict_from_point(self, image_rgb, point_xy) -> Sam2SegmentationResult:
        if self.backend in {"fallback", "stub_sam2"}:
            mask = self._point_seed_mask(image_rgb, point_xy)
            return Sam2SegmentationResult(mask=mask, score=None, prompt_type="point")

        raise NotImplementedError("当前 SAM2 后端尚未绑定实际 predictor。")

    def predict_from_box(self, image_rgb, box_xyxy) -> Sam2SegmentationResult:
        if self.backend in {"fallback", "stub_sam2"}:
            mask = self._box_seed_mask(image_rgb, box_xyxy)
            return Sam2SegmentationResult(mask=mask, score=None, prompt_type="box")

        raise NotImplementedError("当前 SAM2 后端尚未绑定实际 predictor。")

    def segment(self, image_rgb, point_xy=None, box_xyxy=None):
        if point_xy is not None:
            return self.predict_from_point(image_rgb, point_xy)
        if box_xyxy is not None:
            return self.predict_from_box(image_rgb, box_xyxy)
        raise ValueError("SAM2 分割需要 point_xy 或 box_xyxy 至少一种提示。")
