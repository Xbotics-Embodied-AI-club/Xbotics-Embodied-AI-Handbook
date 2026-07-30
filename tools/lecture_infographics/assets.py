from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class AssetSpec:
    filename: str
    crop: tuple[int, int, int, int] | None = None


ASSETS: dict[str, AssetSpec] = {
    "robot_arm": AssetSpec("robot-arm.png", (280, 10, 1450, 930)),
    "robot_arm_camera": AssetSpec("robot-arm-camera.png"),
    "desktop_pick": AssetSpec("desktop-pick-scene.png"),
    "mobile_manipulator": AssetSpec("mobile-manipulator.png"),
    "humanoid_robot": AssetSpec("humanoid-robot.png"),
}


def load_asset(
    asset_dir: Path,
    name: str,
    max_size: tuple[int, int],
) -> Image.Image:
    try:
        spec = ASSETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown asset: {name}") from exc

    with Image.open(asset_dir / spec.filename) as source:
        image = source.convert("RGB")

    if spec.crop is not None:
        image = image.crop(spec.crop)

    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image
