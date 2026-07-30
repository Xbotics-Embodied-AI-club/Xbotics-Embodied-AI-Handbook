from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .components import (
    draw_arrow,
    draw_card,
    draw_header,
    draw_takeaway,
    fit_text,
    font,
)
from .manifest import FigureSpec
from .theme import THEME


WHITE = THEME.canvas
BLUE = THEME.primary
NAVY = THEME.primary_dark
TEXT = THEME.text
MUTED = THEME.muted
YELLOW = THEME.accent
GREEN = THEME.success


@dataclass(frozen=True)
class LaneLayout:
    box: tuple[int, int, int, int]
    text_x: int
    arrow_x: int


LANE_LAYOUTS = {
    "state_up": LaneLayout((100, 288, 195, 866), text_x=122, arrow_x=170),
    "command_down": LaneLayout((1310, 288, 1405, 866), text_x=1333, arrow_x=1381),
}


def _module(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    *,
    fill: str = WHITE,
) -> None:
    draw_card(draw, box, fill=fill, outline="#93BCE8", radius=15)
    fit_text(draw, label, box, 27, 24, "bold", NAVY)


def _flow_lane(
    draw: ImageDraw.ImageDraw,
    layout: LaneLayout,
    label: str,
    *,
    upward: bool,
    color: str,
    fill: str,
) -> None:
    x1, y1, x2, y2 = layout.box
    draw.rounded_rectangle(layout.box, radius=20, fill=fill, outline=color, width=3)
    divider_x = (layout.text_x + layout.arrow_x) // 2
    draw.line((divider_x, y1 + 22, divider_x, y2 - 22), fill=color, width=2)
    start = (layout.arrow_x, y2 - 42) if upward else (layout.arrow_x, y1 + 42)
    end = (layout.arrow_x, y1 + 42) if upward else (layout.arrow_x, y2 - 42)
    draw_arrow(draw, start, end, color=color, width=8)
    vertical_label = "\n".join(label)
    draw.multiline_text(
        (layout.text_x, (y1 + y2) // 2),
        vertical_label,
        font=font(25, "bold"),
        fill=color,
        anchor="mm",
        align="center",
        spacing=10,
    )


def draw_2_2(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    spec: FigureSpec,
    asset_dir: Path,
) -> None:
    del image, asset_dir
    layers = (
        (
            (90, 270, 1405, 460),
            "应用与智能层",
            "#EEF6FF",
            spec.labels[0:4],
            2,
        ),
        (
            (90, 478, 1405, 668),
            "系统与控制层",
            "#FFF8DE",
            spec.labels[4:7],
            3,
        ),
        (
            (90, 686, 1405, 884),
            "硬件与执行层",
            "#EEF9F4",
            spec.labels[7:13],
            3,
        ),
    )
    for (x1, y1, x2, y2), layer_name, fill, labels, columns in layers:
        draw_card(draw, (x1, y1, x2, y2), fill=fill, outline="#79ABE3", radius=20)
        draw.rounded_rectangle((200, y1 + 18, 425, y2 - 18), radius=17, fill=BLUE)
        fit_text(
            draw,
            layer_name,
            (216, y1 + 28, 409, y2 - 28),
            30,
            28,
            "bold",
            WHITE,
        )

        content_x1, content_x2 = 455, 1305
        rows = (len(labels) + columns - 1) // columns
        gap_x, gap_y = 14, 12
        cell_width = (content_x2 - content_x1 - gap_x * (columns - 1)) // columns
        cell_height = (y2 - y1 - 36 - gap_y * (rows - 1)) // rows
        for index, label in enumerate(labels):
            row, col = divmod(index, columns)
            left = content_x1 + col * (cell_width + gap_x)
            top = y1 + 18 + row * (cell_height + gap_y)
            _module(draw, (left, top, left + cell_width, top + cell_height), label)

    _flow_lane(
        draw,
        LANE_LAYOUTS["state_up"],
        spec.labels[13],
        upward=True,
        color=GREEN,
        fill="#E8F7F1",
    )
    _flow_lane(
        draw,
        LANE_LAYOUTS["command_down"],
        spec.labels[14],
        upward=False,
        color="#B97300",
        fill="#FFF3D1",
    )

    draw_card(draw, (1435, 270, 1830, 884), fill="#F7FBFF", outline=BLUE, width=3, radius=22)
    draw.rounded_rectangle((1453, 288, 1812, 364), radius=16, fill=BLUE)
    fit_text(draw, "接口契约必须写清", (1463, 298, 1802, 354), 30, 28, "bold", WHITE)
    for index, item in enumerate(spec.labels[15:], start=1):
        top = 391 + (index - 1) * 77
        draw.ellipse((1462, top, 1512, top + 50), fill=YELLOW, outline="#D49A00", width=2)
        draw.text((1487, top + 25), str(index), font=font(25, "bold"), fill=NAVY, anchor="mm")
        fit_text(draw, item, (1522, top - 3, 1805, top + 53), 27, 25, "bold", TEXT, anchor="lm")


def _topic_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.rounded_rectangle((x - 18, y - 31, x + 18, y + 31), radius=7, fill=WHITE, outline=color, width=4)
    for node_x in (x - 62, x + 62):
        draw.ellipse((node_x - 13, y - 13, node_x + 13, y + 13), fill=WHITE, outline=color, width=4)
    draw_arrow(draw, (x - 47, y), (x - 20, y), color=color, width=5)
    draw_arrow(draw, (x + 20, y), (x + 47, y), color=color, width=5)


def _service_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    for offset in (-48, 48):
        draw.rounded_rectangle((x + offset - 27, y - 26, x + offset + 27, y + 26), radius=9, fill=WHITE, outline=color, width=4)
    draw_arrow(draw, (x - 18, y - 14), (x + 18, y - 14), color=color, width=5)
    draw_arrow(draw, (x + 18, y + 14), (x - 18, y + 14), color=color, width=5)


def _action_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.ellipse((x - 42, y - 42, x + 42, y + 42), outline=color, width=4)
    draw.ellipse((x - 24, y - 24, x + 24, y + 24), outline=color, width=4)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=YELLOW, outline=color, width=2)
    draw.line((x + 30, y - 36, x + 54, y - 49), fill=color, width=5)
    draw.polygon(((x + 62, y - 54), (x + 52, y - 35), (x + 44, y - 50)), fill=color)
    draw.rounded_rectangle((x + 42, y + 24, x + 66, y + 48), radius=4, fill=WHITE, outline=color, width=4)


def _launch_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str) -> None:
    x, y = center
    draw.polygon(((x - 56, y - 27), (x - 56, y + 27), (x - 18, y)), fill=YELLOW, outline=color)
    draw.line((x - 8, y, x + 12, y), fill=color, width=5)
    draw.line((x + 12, y - 30, x + 12, y + 30), fill=color, width=5)
    for offset in (-34, 0, 34):
        draw.line((x + 12, y + offset, x + 40, y + offset), fill=color, width=4)
        draw.ellipse((x + 40, y + offset - 12, x + 64, y + offset + 12), fill=WHITE, outline=color, width=4)


def _mechanism_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    kind: str,
    *,
    fill: str,
    color: str,
) -> None:
    x1, y1, x2, y2 = box
    draw_card(draw, box, fill=fill, outline=color, width=3, radius=24)
    center = ((x1 + x2) // 2, y1 + 100)
    {
        "topic": _topic_icon,
        "service": _service_icon,
        "action": _action_icon,
        "launch": _launch_icon,
    }[kind](draw, center, color)
    draw.line((x1 + 38, y1 + 180, x2 - 38, y1 + 180), fill=color, width=3)
    fit_text(draw, label, (x1 + 20, y1 + 194, x2 - 20, y2 - 18), 28, 24, "bold", NAVY)


def draw_2_4(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    spec: FigureSpec,
    asset_dir: Path,
) -> None:
    del image, asset_dir
    cards = (
        ((90, 275, 505, 575), "topic", "#EAF4FF", BLUE),
        ((530, 275, 945, 575), "service", "#ECF9F4", GREEN),
        ((970, 275, 1385, 575), "action", "#FFF7DD", "#9A6B00"),
        ((1410, 275, 1825, 575), "launch", "#F1F5FA", NAVY),
    )
    for label, (box, kind, fill, color) in zip(spec.labels[:4], cards):
        _mechanism_card(draw, box, label, kind, fill=fill, color=color)

    draw_card(draw, (150, 625, 1770, 884), fill="#FFF8F8", outline=THEME.danger, width=3, radius=24)
    draw.text(
        (960, 684),
        "通信连通 ≠ 系统正确",
        font=font(36, "bold"),
        fill=THEME.danger,
        anchor="mm",
    )
    draw.text((960, 728), "还必须沿契约逐项验证", font=font(25), fill=MUTED, anchor="mm")

    checks = spec.labels[5:]
    start_x, item_width, gap = 260, 280, 105
    for index, item in enumerate(checks):
        left = start_x + index * (item_width + gap)
        draw_card(draw, (left, 770, left + item_width, 850), fill=WHITE, outline="#DE9C9C", radius=17)
        fit_text(draw, item, (left + 12, 780, left + item_width - 12, 840), 28, 26, "bold", TEXT)
        if index < len(checks) - 1:
            draw_arrow(
                draw,
                (left + item_width + 18, 810),
                (left + item_width + gap - 18, 810),
                color=THEME.danger,
                width=6,
            )


C_LAYOUTS = {
    "2-2": draw_2_2,
    "2-4": draw_2_4,
}


def render_c(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw_header(draw, spec)
    C_LAYOUTS[spec.key](image, draw, spec, asset_dir)
    draw_takeaway(draw, spec.takeaway)
