# depth_sources.py
# 功能：深度数据输入源抽象层
# - 定义 DepthFrame 数据结构（RGB、深度图、内参、时间戳等）
# - FileDepthSource：从本地目录加载 depth.png + rgb.png + intrinsics.json
# - UsbDepthCameraSource：通过 pyrealsense2 驱动 RealSense USB 深度相机
# - NetworkDepthSource：通过 JSON manifest 清单加载网络传输的深度数据
# - create_depth_source()：工厂函数，根据类型统一创建输入源实例
#
# 依赖：numpy, Pillow, 可选 pyrealsense2

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class DepthFrame:
    rgb: Optional[np.ndarray]
    depth: np.ndarray
    intrinsics: Dict[str, float]
    depth_scale: float
    timestamp: float
    source_name: str
    metadata: Dict[str, object]


class BaseDepthSource:
    def get_frame(self) -> DepthFrame:
        raise NotImplementedError

    def close(self):
        return None


class FileDepthSource(BaseDepthSource):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.paths = {
            "rgb": os.path.join(base_dir, "rgb.png"),
            "depth": os.path.join(base_dir, "depth.png"),
            "intrinsics": os.path.join(base_dir, "intrinsics.json"),
            "preview": os.path.join(base_dir, "depth_preview.png"),
        }

    def _load_rgb(self):
        for candidate in (self.paths["rgb"], self.paths["preview"]):
            if os.path.exists(candidate):
                return np.asarray(Image.open(candidate).convert("RGB"))
        return None

    def get_frame(self) -> DepthFrame:
        if not os.path.exists(self.paths["depth"]):
            raise FileNotFoundError(f"未找到深度图文件: {self.paths['depth']}")
        if not os.path.exists(self.paths["intrinsics"]):
            raise FileNotFoundError(f"未找到内参文件: {self.paths['intrinsics']}")

        with open(self.paths["intrinsics"], "r", encoding="utf-8") as fp:
            config = json.load(fp)

        depth = np.asarray(Image.open(self.paths["depth"]))
        if depth.ndim != 2:
            raise ValueError("文件输入源的 depth.png 必须是单通道深度图。")

        rgb = self._load_rgb()
        intrinsics = {
            "fx": float(config["fx"]),
            "fy": float(config["fy"]),
            "cx": float(config["cx"]),
            "cy": float(config["cy"]),
        }
        depth_scale = float(config.get("depth_scale", 1000.0))

        return DepthFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            timestamp=time.time(),
            source_name="file",
            metadata={"base_dir": self.base_dir},
        )


class UsbDepthCameraSource(BaseDepthSource):
    def __init__(self, device: str = "auto", align_to_color: bool = True):
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 pyrealsense2，无法使用 USB 深度相机输入。"
            ) from exc

        self.rs = rs
        self.device = device
        self.align_to_color = align_to_color
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color) if align_to_color else None

        if device != "auto":
            self.config.enable_device(device)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

    def get_frame(self) -> DepthFrame:
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("相机未返回完整的 RGB-D 帧。")

        depth = np.asarray(depth_frame.get_data())
        rgb = np.asarray(color_frame.get_data())

        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        intrinsics_dict = {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
        }

        return DepthFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics_dict,
            depth_scale=1.0 / self.depth_scale if self.depth_scale > 0 else 1000.0,
            timestamp=time.time(),
            source_name="usb_realsense",
            metadata={
                "device": self.device,
                "align_to_color": self.align_to_color,
                "native_depth_scale_m": self.depth_scale,
            },
        )

    def close(self):
        self.pipeline.stop()


class NetworkDepthSource(BaseDepthSource):
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path

    def get_frame(self) -> DepthFrame:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"未找到网络输入清单文件: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        base_dir = os.path.dirname(self.manifest_path)

        depth_path = os.path.join(base_dir, manifest["depth"])
        rgb_path = os.path.join(base_dir, manifest["rgb"]) if manifest.get("rgb") else None

        intrinsics = manifest["intrinsics"]
        depth_scale = float(manifest.get("depth_scale", 1000.0))

        depth = np.asarray(Image.open(depth_path))
        if depth.ndim != 2:
            raise ValueError("网络输入 manifest 指向的 depth 文件必须是单通道深度图。")

        rgb = None
        if rgb_path and os.path.exists(rgb_path):
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))

        return DepthFrame(
            rgb=rgb,
            depth=depth,
            intrinsics={
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
            },
            depth_scale=depth_scale,
            timestamp=time.time(),
            source_name="network_manifest",
            metadata={"manifest_path": self.manifest_path},
        )


def create_depth_source(source_type, base_dir=None, manifest_path=None, device="auto"):
    if source_type == "file":
        if not base_dir:
            raise ValueError("file 输入源需要提供 base_dir。")
        return FileDepthSource(base_dir)

    if source_type == "usb":
        return UsbDepthCameraSource(device=device)

    if source_type == "network":
        if not manifest_path:
            raise ValueError("network 输入源需要提供 manifest_path。")
        return NetworkDepthSource(manifest_path)

    raise ValueError(f"不支持的输入源类型: {source_type}")
