# interactive_mask_ui.py
# 功能：基于 OpenCV 的交互式掩膜选择 UI
# - InteractiveMaskSelector：提供鼠标交互窗口
#   - 单击 → 点提示 (point prompt)
#   - 拖拽 → 框提示 (box prompt)
#   - Enter 确认、R 重置、Esc 取消
# - preview_mask_overlay()：分割结果预览，支持接受/重选/取消
# - InteractionResult：封装交互结果的数据类
#
# 依赖：opencv-python, numpy

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未安装 opencv-python，请先执行 `pip install opencv-python` 或在已安装 OpenCV 的环境中运行。"
    ) from exc

import numpy as np


@dataclass
class InteractionResult:
    mode: str
    point: Optional[Tuple[int, int]] = None
    box_xyxy: Optional[Tuple[int, int, int, int]] = None


class InteractiveMaskSelector:
    def __init__(self, window_name="Interactive Target Selector"):
        self.window_name = window_name
        self._base_image = None
        self._current_image = None
        self._point = None
        self._drag_start = None
        self._drag_end = None
        self._box = None
        self._instruction_image = None

    @staticmethod
    def _to_bgr(image):
        array = np.asarray(image)
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    def _refresh_canvas(self):
        canvas = self._base_image.copy()
        if self._point is not None:
            cv2.circle(canvas, self._point, 6, (0, 0, 255), -1)
            cv2.circle(canvas, self._point, 18, (0, 255, 255), 2)

        if self._drag_start is not None and self._drag_end is not None:
            cv2.rectangle(canvas, self._drag_start, self._drag_end, (0, 255, 255), 2)
        elif self._box is not None:
            x1, y1, x2, y2 = self._box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(canvas, "Click: point prompt | Drag: box prompt | Enter: confirm | R: reset | Esc: cancel",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 60), 2, cv2.LINE_AA)
        self._current_image = canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_end = (x, y)
            self._refresh_canvas()
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            start_x, start_y = self._drag_start
            end_x, end_y = x, y
            self._drag_end = (end_x, end_y)

            if abs(end_x - start_x) < 5 and abs(end_y - start_y) < 5:
                self._point = (x, y)
                self._box = None
            else:
                x1, x2 = sorted([start_x, end_x])
                y1, y2 = sorted([start_y, end_y])
                self._box = (x1, y1, x2, y2)
                self._point = None

            self._drag_start = None
            self._drag_end = None
            self._refresh_canvas()

    def select(self, image_rgb) -> Optional[InteractionResult]:
        self._base_image = self._to_bgr(image_rgb)
        self._point = None
        self._drag_start = None
        self._drag_end = None
        self._box = None
        self._refresh_canvas()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        try:
            while True:
                cv2.imshow(self.window_name, self._current_image)
                key = cv2.waitKey(20) & 0xFF

                if key in (13, 10):  # Enter
                    if self._point is not None:
                        return InteractionResult(mode="point", point=self._point)
                    if self._box is not None:
                        return InteractionResult(mode="box", box_xyxy=self._box)
                elif key in (ord('r'), ord('R')):
                    self._point = None
                    self._box = None
                    self._drag_start = None
                    self._drag_end = None
                    self._refresh_canvas()
                elif key == 27:  # ESC
                    return None
        finally:
            cv2.destroyWindow(self.window_name)


def preview_mask_overlay(image_rgb, mask, window_name="Mask Preview"):
    image_bgr = InteractiveMaskSelector._to_bgr(image_rgb)
    mask_bool = np.asarray(mask).astype(bool)

    overlay = image_bgr.copy()
    overlay[mask_bool] = (0.35 * overlay[mask_bool] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(overlay, "Enter: accept | R: reselect | Esc: cancel",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 60), 2, cv2.LINE_AA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                return "accept"
            if key in (ord('r'), ord('R')):
                return "reselect"
            if key == 27:
                return "cancel"
    finally:
        cv2.destroyWindow(window_name)
