from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageCms, ImageDraw

from .layouts_a import render_a
from .layouts_b import render_b
from .layouts_c import render_c
from .manifest import FigureSpec
from .theme import THEME


W, H = 1920, 1080
WHITE = THEME.canvas
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def render_figure(spec: FigureSpec, asset_dir: Path, output: Path) -> None:
    image = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(image)
    if spec.template == "A":
        render_a(image, draw, spec, asset_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
        return
    if spec.template == "B":
        render_b(image, draw, spec, asset_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
        return
    render_c(image, draw, spec, asset_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
