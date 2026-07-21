from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageCms, ImageDraw

from .components import (
    draw_arrow as arrow,
    draw_card as rounded,
    draw_header as header,
    draw_takeaway as takeaway,
    fit_text,
    font,
)
from .layouts_a import render_a
from .manifest import FigureSpec
from .theme import THEME


W, H = 1920, 1080
BLUE = THEME.primary
NAVY = THEME.primary_dark
YELLOW = THEME.accent
TEXT = THEME.text
MUTED = THEME.muted
WHITE = THEME.canvas
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text_value: str,
         fill: str = WHITE, outline: str = "#8CB9EC", text_fill: str = TEXT,
         size: int = 26, bold: bool = True) -> None:
    rounded(draw, box, fill=fill, outline=outline, width=2, radius=16)
    fit_text(draw, text_value, box, size, 24, bold, text_fill)


def visual_2_2(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    # Three explicit layers plus a full-height interface contract panel.
    x1, x2 = 105, 1390
    layers = [
        (285, 450, "应用与智能层", "#EEF6FF", ("任务应用", "规划与策略", "感知与状态估计", "数据与运维")),
        (475, 655, "系统与控制层", "#FFF8DE", ("ROS2 / 消息接口", "控制与安全", "硬件抽象")),
        (680, 875, "硬件与执行层", "#EEF9F4", ("计算平台", "传感器", "执行器", "机器人本体", "通信", "电源与急停")),
    ]
    for top, bottom, name, fill, items in layers:
        rounded(draw, (x1, top, x2, bottom), fill=fill, outline="#7AA9DF")
        draw.rounded_rectangle((x1 + 16, top + 16, x1 + 235, bottom - 16), radius=15, fill=BLUE)
        fit_text(draw, name, (x1 + 28, top + 20, x1 + 223, bottom - 20), 28, 24, True, WHITE)
        available = x2 - (x1 + 275) - 30
        gap = 16
        item_w = (available - gap * (len(items) - 1)) // len(items)
        for i, item in enumerate(items):
            ix = x1 + 265 + i * (item_w + gap)
            pill(draw, (ix, top + 32, ix + item_w, bottom - 32), item, size=24)
    arrow(draw, (352, 852), (352, 310), color="#1A9B67", width=8)
    draw.text((352, 254), "状态上行 ↑", font=font(24, True), fill="#147A58", anchor="mm")
    arrow(draw, (1360, 310), (1360, 852), color="#E08A1E", width=8)
    draw.text((1280, 254), "指令下行 ↓", font=font(24, True), fill="#B66B0D", anchor="mm")
    rounded(draw, (1430, 285, 1815, 875), fill="#F8FBFF", outline=BLUE, width=3)
    draw.rounded_rectangle((1448, 303, 1797, 370), radius=15, fill=BLUE)
    draw.text((1622, 337), "接口契约必须写清", font=font(28, True), fill=WHITE, anchor="mm")
    contract = ("含义", "维度与顺序", "单位与坐标系", "时间戳与频率", "范围", "异常行为")
    for i, item in enumerate(contract):
        y = 395 + i * 73
        draw.ellipse((1462, y + 10, 1494, y + 42), fill=YELLOW)
        draw.text(
            (1478, y + 27),
            str(i + 1),
            font=font(THEME.small_size, "bold"),
            fill=NAVY,
            anchor="mm",
        )
        draw.text((1510, y + 27), item, font=font(25, True), fill=TEXT, anchor="lm")


def visual_2_4(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    mechanisms = spec.labels[:4]
    colors = ("#EAF5FF", "#EDF9F4", "#FFF7DD", "#F4EEFF")
    for i, (label, fill) in enumerate(zip(mechanisms, colors)):
        x = 100 + i * 435
        rounded(draw, (x, 315, x + 380, 555), fill=fill, outline="#78A9DF")
        draw.ellipse((x + 135, 340, x + 245, 450), fill=WHITE, outline=BLUE, width=4)
        draw.text((x + 190, 395), str(i + 1), font=font(39, True), fill=BLUE, anchor="mm")
        fit_text(draw, label, (x + 15, 465, x + 365, 535), 26, 24, True)
    rounded(draw, (180, 625, 1740, 845), fill="#FFF7F7", outline="#DB7777")
    draw.text((400, 680), "通信连通 ≠ 系统正确", font=font(36, True), fill="#B33F3F", anchor="mm")
    arrow(draw, (630, 730), (760, 730), color="#D95B5B", width=7)
    checks = spec.labels[5:]
    for i, item in enumerate(checks):
        x = 790 + i * 220
        pill(draw, (x, 675, x + 185, 790), item, fill=WHITE, outline="#D8A0A0", text_fill="#8B3D3D", size=26)


def generic_b(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    steps = spec.steps
    count = len(steps)
    cols = 4 if count > 6 else 3
    rows = (count + cols - 1) // cols
    card_w = 450 if cols == 3 else 330
    gap = 50
    total_w = cols * card_w + (cols - 1) * gap
    left = (W - total_w) // 2
    card_h = 190 if rows <= 2 else 145
    top = 292
    for index, label in enumerate(steps):
        row, col = divmod(index, cols)
        x, y = left + col * (card_w + gap), top + row * (card_h + 54)
        rounded(draw, (x, y, x + card_w, y + card_h), fill="#F8FBFF", outline="#7BB0EC")
        draw.rounded_rectangle((x + 18, y + 18, x + 78, y + 78), radius=14, fill=BLUE)
        draw.text((x + 48, y + 49), str(index + 1), font=font(32, True), fill=WHITE, anchor="mm")
        fit_text(draw, label, (x + 92, y + 25, x + card_w - 18, y + card_h - 20), 33, 24, True)
        if col < cols - 1 and index + 1 < count:
            arrow(draw, (x + card_w + 9, y + card_h // 2), (x + card_w + gap - 9, y + card_h // 2), width=6)
        elif row < rows - 1 and index + 1 < count:
            arrow(draw, (x + card_w // 2, y + card_h + 8), (x + card_w // 2, y + card_h + 43), width=6)
    if spec.labels:
        fit_text(draw, " · ".join(spec.labels), (260, 852, 1660, 914), 25, 24, False, MUTED)


def generic_c(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    labels = spec.labels
    groups = [labels[i::3] for i in range(3)]
    titles = ("软件与应用", "接口与控制", "硬件与约束")
    fills = ("#EEF6FF", "#FFF8DF", "#F1FAF7")
    for col in range(3):
        x = 105 + col * 575
        rounded(draw, (x, 275, x + 520, 870), fill=fills[col], outline="#75A7DF")
        draw.rounded_rectangle((x + 18, 292, x + 502, 352), radius=14, fill=BLUE)
        draw.text((x + 260, 322), titles[col], font=font(29, True), fill=WHITE, anchor="mm")
        for row, label in enumerate(groups[col][:6]):
            y = 380 + row * 76
            rounded(draw, (x + 34, y, x + 486, y + 58), fill=WHITE, outline="#B5D2F2", width=2, radius=13)
            fit_text(draw, label, (x + 45, y + 4, x + 475, y + 54), 25, 24, row == 0)


def render_figure(spec: FigureSpec, asset_dir: Path, output: Path) -> None:
    image = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(image)
    if spec.template == "A":
        render_a(image, draw, spec, asset_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
        return
    header(draw, spec)
    custom = {
        "2-2": visual_2_2,
        "2-4": visual_2_4,
    }.get(spec.key)
    if custom:
        custom(draw, spec)
    elif spec.template == "B":
        generic_b(draw, spec)
    else:
        generic_c(draw, spec)
    takeaway(draw, spec.takeaway)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
