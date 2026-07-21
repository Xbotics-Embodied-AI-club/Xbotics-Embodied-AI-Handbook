from __future__ import annotations

from math import hypot
from typing import TypeAlias

from PIL import Image, ImageDraw, ImageFont

from .manifest import FigureSpec
from .theme import THEME


Box: TypeAlias = tuple[int, int, int, int]
Point: TypeAlias = tuple[int, int]

FONT_REGULAR = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"


def _weight_name(weight: str | bool) -> str:
    if isinstance(weight, bool):
        return "bold" if weight else "regular"
    if weight not in {"regular", "bold"}:
        raise ValueError(f"unsupported font weight: {weight}")
    return weight


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if _weight_name(weight) == "bold" else FONT_REGULAR
    return ImageFont.truetype(path, size)


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    max_size: int,
    min_size: int = 24,
    weight: str = "regular",
    fill: str | None = None,
    anchor: str = "mm",
) -> int:
    if min_size > max_size:
        raise ValueError("min_size must not exceed max_size")

    x1, y1, x2, y2 = box
    available_width = x2 - x1 - 20
    available_height = y2 - y1 - 16
    for size in range(max_size, min_size - 1, -1):
        chosen_font = font(size, _weight_name(weight))
        bounds = draw.multiline_textbbox(
            (0, 0), text, font=chosen_font, spacing=8, align="center"
        )
        if (
            bounds[2] - bounds[0] <= available_width
            and bounds[3] - bounds[1] <= available_height
        ):
            draw.multiline_text(
                ((x1 + x2) // 2, (y1 + y2) // 2),
                text,
                font=chosen_font,
                fill=fill or THEME.text,
                anchor=anchor,
                align="center",
                spacing=8,
            )
            return size

    raise ValueError(f"text cannot fit box at minimum size {min_size}: {text!r}")


def _draw_major_shadow(draw: ImageDraw.ImageDraw, box: Box, radius: int) -> None:
    image = draw._image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = box
    shadow.rounded_rectangle(
        (x1, y1 + 6, x2, y2 + 6),
        radius=radius,
        fill=(82, 111, 142, 38),
    )
    if image.mode == "RGBA":
        image.alpha_composite(overlay)
    else:
        image.paste(overlay, (0, 0), overlay)


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: Box,
    style: str = "default",
    *,
    fill: str | None = None,
    outline: str | None = None,
    width: int | None = None,
    radius: int | None = None,
) -> None:
    styles = {
        "default": (THEME.canvas, THEME.border),
        "major": (THEME.canvas, THEME.border),
        "soft": (THEME.primary_soft, THEME.border),
        "success": ("#ECF9F4", THEME.success),
        "danger": ("#FFF0F0", THEME.danger),
    }
    if style not in styles:
        raise ValueError(f"unknown card style: {style}")
    default_fill, default_outline = styles[style]
    chosen_radius = radius if radius is not None else THEME.card_radius
    if style == "major":
        _draw_major_shadow(draw, box, chosen_radius)
    draw.rounded_rectangle(
        box,
        radius=chosen_radius,
        fill=fill or default_fill,
        outline=outline or default_outline,
        width=width if width is not None else THEME.card_border,
    )


def draw_header(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    draw.text(
        (960, 76),
        spec.title,
        font=font(THEME.title_size, "bold"),
        fill=THEME.primary,
        anchor="mm",
    )
    draw.rounded_rectangle((900, 120, 1020, 129), radius=5, fill=THEME.accent)
    draw.text(
        (960, 166),
        spec.subtitle,
        font=font(THEME.subtitle_size),
        fill=THEME.primary_dark,
        anchor="mm",
    )
    draw_card(
        draw,
        (52, 204, 1868, 934),
        style="major",
        outline="#62A1EA",
        radius=28,
    )


def draw_takeaway(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw_card(
        draw,
        (230, 960, 1690, 1042),
        outline=THEME.primary,
        radius=20,
    )
    draw.rounded_rectangle((250, 976, 306, 1027), radius=12, fill=THEME.primary)
    draw.ellipse((267, 993, 289, 1015), outline=THEME.canvas, width=4)
    draw.line((278, 983, 278, 1024), fill=THEME.canvas, width=3)
    draw.line((258, 1004, 298, 1004), fill=THEME.canvas, width=3)
    fit_text(
        draw,
        text,
        (325, 970, 1670, 1033),
        THEME.subtitle_size,
        THEME.min_text_size,
        "bold",
        THEME.primary,
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    semantic: str = "primary",
    width: int | None = None,
    *,
    color: str | None = None,
) -> None:
    colors = {
        "primary": THEME.primary,
        "secondary": THEME.border,
        "success": THEME.success,
        "danger": THEME.danger,
        "accent": THEME.accent,
    }
    chosen_color = color or colors.get(semantic, semantic)
    chosen_width = width
    if chosen_width is None:
        chosen_width = (
            THEME.secondary_arrow if semantic == "secondary" else THEME.primary_arrow
        )
    draw.line((*start, *end), fill=chosen_color, width=chosen_width)

    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    length = max(hypot(dx, dy), 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    arrowhead = 20
    points = [
        (ex, ey),
        (
            ex - ux * arrowhead + px * arrowhead * 0.55,
            ey - uy * arrowhead + py * arrowhead * 0.55,
        ),
        (
            ex - ux * arrowhead - px * arrowhead * 0.55,
            ey - uy * arrowhead - py * arrowhead * 0.55,
        ),
    ]
    draw.polygon(points, fill=chosen_color)
