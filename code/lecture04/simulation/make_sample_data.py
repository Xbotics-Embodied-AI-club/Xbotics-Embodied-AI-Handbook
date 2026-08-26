"""make_sample_data.py —— 第 4 讲 Demo（无硬件仿真版）样例数据合成。

程序化生成一对「已对齐」的 RGB 图与深度图（uint16 毫米），以及一个已知真值的
物块，供反投影 / 坐标变换 / 误差扰动实验使用。零外部下载、确定性可复现。

设计目标（与讲义 4.6.6 / 4.10.3 对齐）：
- 物块顶面中心在相机坐标系下的真值  point_camera_gt = [0.10, -0.02, 0.60] m
- 用内参投影后落在像素 (u, v) = (420, 220)
- 物块顶面深度 600 mm，背景深度 1200 mm
- 物块中心附近故意制造几个无效深度（0，模拟反光/空洞），用于演示邻域中位数过滤

用法：
    python make_sample_data.py           # 生成 data/rgb.png 与 data/depth.npy
    python make_sample_data.py --show    # 额外弹窗显示合成结果（需 GUI）

也可作为模块导入：`from make_sample_data import synthesize_scene, load_intrinsics`
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# 相机内参（与 camera_info.yaml / 讲义 4.6.3 一致）
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX, FY = 600.0, 600.0
CX, CY = 320.0, 240.0

# ---------------------------------------------------------------------------
# 场景真值（与讲义 4.6.6 的例子完全一致）
# ---------------------------------------------------------------------------
# 相机光心在 base 系中的安装位置：前方 0.20 m、左侧 0.05 m、上方 0.40 m
T_BASE_CAMERA_T = np.array([0.20, 0.05, 0.40])
# optical frame -> base 的旋转（REP-103：camera_optical z 向前 / x 向右 / y 向下）
R_BASE_OPTICAL = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]]
)
# 物块顶面中心在相机坐标系下的真值（米）
POINT_CAMERA_GT = np.array([0.10, -0.02, 0.60])

# 场景深度（毫米）
BACKGROUND_DEPTH_MM = 1200.0
BLOCK_DEPTH_MM = 600.0

# 物块顶面矩形区域（像素，半开区间 [x0, x1) x [y0, y1)），中心为投影像素
BLOCK_X = (370, 470)
BLOCK_Y = (180, 260)

# 无效深度空洞（像素坐标，模拟反光/空洞，值置 0）
INVALID_PIXELS = [(420, 220), (421, 219), (419, 221)]


def project(camera_point: np.ndarray) -> tuple[int, int]:
    """针孔投影：camera 系三维点 -> 像素 (u, v)，返回整数像素。"""
    X, Y, Z = camera_point
    u = FX * X / Z + CX
    v = FY * Y / Z + CY
    return int(round(u)), int(round(v))


def load_intrinsics(path: str | None = None) -> dict:
    """从 camera_info.yaml 读取内参（返回 dict，与 notebook 保持一致）。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "camera_info.yaml")
    with open(path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    k = np.array(info["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    return {
        "width": int(info["image_width"]),
        "height": int(info["image_height"]),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def synthesize_scene(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """合成 RGB 图 (H, W, 3) uint8 与深度图 (H, W) uint16（毫米）。

    深度图单位刻意用毫米：对应讲义 4.6.4「depth 的单位是米、毫米还是 depth unit」
    以及 4.10.5「深度单位与 scale——毫米是否被当成米」的经典错误场景。
    """
    rng = np.random.default_rng(seed)

    # --- 深度图：背景 -> 物块顶面 -> 无效空洞 ---
    depth_mm = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_DEPTH_MM, dtype=np.float64)
    depth_mm[BLOCK_Y[0]:BLOCK_Y[1], BLOCK_X[0]:BLOCK_X[1]] = BLOCK_DEPTH_MM
    for (u, v) in INVALID_PIXELS:
        depth_mm[v, u] = 0.0  # 0 表示无效深度（讲义 4.6.4）

    # --- RGB 图：灰色桌面 + 黄色物块 + 中心十字标记 ---
    rgb = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 150, dtype=np.uint8)  # 浅灰桌面
    rgb[BLOCK_Y[0]:BLOCK_Y[1], BLOCK_X[0]:BLOCK_X[1]] = (240, 200, 0)  # 黄色物块

    # 物块边缘描一圈深色边框，便于识别矩形区域
    rgb[BLOCK_Y[0]:BLOCK_Y[0] + 2, BLOCK_X[0]:BLOCK_X[1]] = (60, 60, 60)
    rgb[BLOCK_Y[1] - 2:BLOCK_Y[1], BLOCK_X[0]:BLOCK_X[1]] = (60, 60, 60)
    rgb[BLOCK_Y[0]:BLOCK_Y[1], BLOCK_X[0]:BLOCK_X[0] + 2] = (60, 60, 60)
    rgb[BLOCK_Y[0]:BLOCK_Y[1], BLOCK_X[1] - 2:BLOCK_X[1]] = (60, 60, 60)

    # 中心黑色十字标记（提示“点击这里”的目标像素）
    uc, vc = project(POINT_CAMERA_GT)
    half = 4
    rgb[vc - half:vc + half + 1, uc - 1:uc + 2] = (0, 0, 0)
    rgb[vc - 1:vc + 2, uc - half:uc + half + 1] = (0, 0, 0)

    # 轻量确定性噪声，让图像不至于完全平滑（不影响真值深度）
    noise = rng.integers(0, 8, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return rgb, depth_mm.astype(np.uint16)


def save_sample_data(data_dir: str = "data") -> dict:
    """生成并保存样例数据，返回真值信息（供打印 / notebook 对照）。"""
    os.makedirs(data_dir, exist_ok=True)
    rgb, depth_mm = synthesize_scene()

    rgb_path = os.path.join(data_dir, "rgb.png")
    depth_path = os.path.join(data_dir, "depth.npy")

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth_mm)  # 保留 uint16 毫米，读取方需 /1000 转米

    uc, vc = project(POINT_CAMERA_GT)
    point_base_gt = R_BASE_OPTICAL @ POINT_CAMERA_GT + T_BASE_CAMERA_T

    return {
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "depth_dtype": str(depth_mm.dtype),
        "depth_min_mm": int(depth_mm[depth_mm > 0].min()),
        "depth_max_mm": int(depth_mm.max()),
        "block_center_pixel": (uc, vc),
        "point_camera_gt": POINT_CAMERA_GT,
        "point_base_gt": point_base_gt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="合成第 4 讲 Demo 样例数据")
    ap.add_argument("--show", action="store_true", help="弹窗显示合成结果")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    info = save_sample_data(os.path.join(here, "data"))

    print("样例数据已生成：")
    for k, v in info.items():
        print(f"  {k:20s} = {v}")

    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(Image.open(info["rgb_path"]))
        ax[0].set_title("RGB")
        ax[1].imshow(np.load(info["depth_path"]), cmap="plasma")
        ax[1].set_title("Depth (mm, uint16)")
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
