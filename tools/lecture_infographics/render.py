from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageCms, ImageDraw, ImageFont

from .manifest import FigureSpec


W, H = 1920, 1080
BLUE = "#0759C7"
NAVY = "#123463"
CYAN = "#EAF5FF"
YELLOW = "#FFC928"
TEXT = "#24364B"
MUTED = "#61738A"
WHITE = "#FFFFFF"
FONT_REGULAR = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
            fill: str = WHITE, outline: str = "#B9D7F8", width: int = 3,
            radius: int = 24) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int],
             max_size: int, min_size: int = 24, bold: bool = False,
             fill: str = TEXT, anchor: str = "mm") -> None:
    x1, y1, x2, y2 = box
    for size in range(max_size, min_size - 1, -2):
        f = font(size, bold)
        bounds = draw.multiline_textbbox((0, 0), text, font=f, spacing=8, align="center")
        if bounds[2] - bounds[0] <= x2 - x1 - 20 and bounds[3] - bounds[1] <= y2 - y1 - 16:
            draw.multiline_text(((x1 + x2) // 2, (y1 + y2) // 2), text, font=f,
                                fill=fill, anchor=anchor, align="center", spacing=8)
            return


def header(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    draw.text((960, 76), spec.title, font=font(60, True), fill=BLUE, anchor="mm")
    draw.rounded_rectangle((900, 120, 1020, 129), radius=5, fill=YELLOW)
    draw.text((960, 166), spec.subtitle, font=font(30), fill=NAVY, anchor="mm")
    rounded(draw, (52, 204, 1868, 934), outline="#62A1EA", width=3, radius=28)


def takeaway(draw: ImageDraw.ImageDraw, text: str) -> None:
    rounded(draw, (230, 960, 1690, 1042), outline=BLUE, width=3, radius=20)
    draw.rounded_rectangle((250, 976, 306, 1027), radius=12, fill=BLUE)
    draw.ellipse((267, 993, 289, 1015), outline=WHITE, width=4)
    draw.line((278, 983, 278, 1024), fill=WHITE, width=3)
    draw.line((258, 1004, 298, 1004), fill=WHITE, width=3)
    fit_text(draw, text, (325, 970, 1670, 1033), 30, 23, True, BLUE)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          color: str = BLUE, width: int = 8) -> None:
    draw.line((*start, *end), fill=color, width=width)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    s = 20
    points = [(ex, ey), (ex - ux * s + px * s * .55, ey - uy * s + py * s * .55),
              (ex - ux * s - px * s * .55, ey - uy * s - py * s * .55)]
    draw.polygon(points, fill=color)


def robot_arm(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0,
              color: str = "#E8EDF3") -> None:
    """Draw a clean, textbook-style six-axis robot arm."""
    def p(dx: int, dy: int) -> tuple[int, int]:
        return x + int(dx * scale), y + int(dy * scale)
    def joint(cx: int, cy: int, r: int) -> None:
        px, py = p(cx, cy)
        rr = int(r * scale)
        draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=WHITE, outline=NAVY, width=max(2, int(4 * scale)))
        draw.ellipse((px - rr // 2, py - rr // 2, px + rr // 2, py + rr // 2), fill=BLUE)
    width = max(12, int(34 * scale))
    draw.rounded_rectangle((*p(-85, 125), *p(85, 170)), radius=int(18 * scale), fill="#CBD5E1", outline=NAVY, width=3)
    draw.line((*p(0, 128), *p(-8, 54)), fill=color, width=width)
    draw.line((*p(-8, 54), *p(58, -28)), fill=color, width=width)
    draw.line((*p(58, -28), *p(130, 20)), fill=color, width=width)
    joint(0, 112, 31); joint(-8, 54, 27); joint(58, -28, 25); joint(130, 20, 22)
    draw.line((*p(130, 20), *p(157, 58)), fill=NAVY, width=max(5, int(10 * scale)))
    draw.line((*p(157, 58), *p(147, 83)), fill=NAVY, width=max(4, int(7 * scale)))
    draw.line((*p(157, 58), *p(170, 80)), fill=NAVY, width=max(4, int(7 * scale)))


def sensor_icon(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "camera":
        rounded(draw, (x - 58, y - 38, x + 58, y + 38), fill="#243B53", outline=NAVY, width=3, radius=12)
        draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill="#72D4FF", outline=WHITE, width=5)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=BLUE)
    elif kind == "compute":
        rounded(draw, (x - 56, y - 48, x + 56, y + 48), fill="#EAF5FF", outline=BLUE, width=4, radius=12)
        for k in range(-2, 3):
            draw.line((x - 72, y + k * 16, x - 56, y + k * 16), fill=BLUE, width=4)
            draw.line((x + 56, y + k * 16, x + 72, y + k * 16), fill=BLUE, width=4)
        draw.rectangle((x - 25, y - 22, x + 25, y + 22), fill=BLUE)
    elif kind == "shield":
        draw.polygon([(x, y - 58), (x + 50, y - 36), (x + 42, y + 28), (x, y + 62), (x - 42, y + 28), (x - 50, y - 36)], fill="#EAF8F2", outline="#1A9B67")
        draw.line((x - 22, y, x - 5, y + 18, x + 27, y - 23), fill="#1A9B67", width=9)
    else:
        draw.ellipse((x - 48, y - 48, x + 48, y + 48), fill=CYAN, outline=BLUE, width=4)
        draw.text((x, y), kind[:1], font=font(38, True), fill=BLUE, anchor="mm")


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text_value: str,
         fill: str = WHITE, outline: str = "#8CB9EC", text_fill: str = TEXT,
         size: int = 26, bold: bool = True) -> None:
    rounded(draw, box, fill=fill, outline=outline, width=2, radius=16)
    fit_text(draw, text_value, box, size, 18, bold, text_fill)


def visual_1_1(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    rounded(draw, (125, 300, 795, 830), fill="#F4F7FB", outline="#B4C8E0")
    rounded(draw, (1125, 300, 1795, 830), fill="#F2FAF7", outline="#84CDB1")
    draw.text((460, 350), "数字 AI", font=font(42, True), fill=NAVY, anchor="mm")
    draw.text((1460, 350), "具身智能", font=font(42, True), fill="#147A58", anchor="mm")
    rounded(draw, (275, 430, 645, 625), fill=WHITE, outline="#86A9CE")
    for i, width in enumerate((240, 190, 220, 145)):
        draw.rounded_rectangle((330, 470 + i * 30, 330 + width, 482 + i * 30), radius=6, fill="#7C9CBE")
    draw.text((460, 690), "输出信息", font=font(32, True), fill=MUTED, anchor="mm")
    robot_arm(draw, 1415, 500, .9)
    draw.rectangle((1260, 730, 1630, 757), fill="#D8E1EA")
    draw.ellipse((1560, 660, 1620, 720), fill=YELLOW, outline="#C79400", width=3)
    arrow(draw, (820, 565), (1090, 565), color=BLUE, width=10)
    draw.text((955, 515), "行动", font=font(31, True), fill=BLUE, anchor="mm")
    draw.arc((815, 600, 1095, 780), 10, 170, fill="#1A9B67", width=7)
    draw.text((955, 780), "环境反馈", font=font(28, True), fill="#147A58", anchor="mm")


def visual_1_2(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    cx, cy = 960, 575
    draw.ellipse((760, 375, 1160, 775), fill="#F2F8FF", outline=BLUE, width=5)
    robot_arm(draw, 915, 470, .62)
    draw.text((960, 720), "具身智能体", font=font(34, True), fill=BLUE, anchor="mm")
    positions = [(360, 350), (620, 300), (1300, 300), (1560, 350), (420, 760), (730, 850), (1490, 760)]
    for index, (label, (x, y)) in enumerate(zip(spec.labels, positions)):
        arrow(draw, (cx + (-1 if x < cx else 1) * 205, cy), (x + (70 if x < cx else -70), y), color="#8AB5E9", width=4)
        draw.ellipse((x - 78, y - 78, x + 78, y + 78), fill=WHITE, outline=BLUE, width=4)
        draw.text((x, y - 9), str(index + 1), font=font(28, True), fill=YELLOW, anchor="mm")
        draw.text((x, y + 30), label, font=font(28, True), fill=NAVY, anchor="mm")


def visual_1_4(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    x_positions = (220, 570, 920, 1270, 1620)
    heights = (215, 285, 355, 425, 495)
    colors = ("#EAF5FF", "#F2F8FF", "#EAF8F2", "#FFF6D7", "#FFF0E9")
    for i, (label, x, h, color) in enumerate(zip(spec.labels, x_positions, heights, colors)):
        y2 = 845
        rounded(draw, (x - 135, y2 - h, x + 135, y2), fill=color, outline=BLUE, radius=24)
        draw.ellipse((x - 48, y2 - h + 35, x + 48, y2 - h + 131), fill=WHITE, outline=BLUE, width=3)
        draw.text((x, y2 - h + 83), str(i + 1), font=font(34, True), fill=BLUE, anchor="mm")
        fit_text(draw, label, (x - 115, y2 - h + 145, x + 115, y2 - 25), 31, 21, True)
        if i:
            arrow(draw, (x - 210, y2 - h // 2), (x - 150, y2 - h // 2), color="#83AEE0", width=5)
    draw.text((960, 300), "可靠执行  →  从示范学习  →  探索优化  →  多模态泛化  →  预测未来",
              font=font(28, True), fill=NAVY, anchor="mm")


def visual_1_5(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    inputs = [(260, 390, "视觉", "camera"), (260, 560, "语言", "L"), (260, 730, "机器人状态", "S")]
    for x, y, label, icon in inputs:
        sensor_icon(draw, x, y, icon)
        pill(draw, (340, y - 43, 600, y + 43), label)
        arrow(draw, (610, y), (760, 570), width=6)
    rounded(draw, (770, 380, 1135, 760), fill="#EEF5FF", outline=BLUE, width=4)
    draw.text((952, 445), "VLA", font=font(62, True), fill=BLUE, anchor="mm")
    draw.text((952, 525), "多模态融合", font=font(32, True), fill=NAVY, anchor="mm")
    draw.text((952, 585), "理解任务与场景", font=font(25), fill=MUTED, anchor="mm")
    draw.text((952, 635), "生成动作建议", font=font(25), fill=MUTED, anchor="mm")
    arrow(draw, (1145, 570), (1270, 570), width=8)
    pill(draw, (1280, 430, 1535, 545), "动作建议", fill="#FFF7DC", outline="#E0B027", size=31)
    arrow(draw, (1408, 560), (1408, 650), color="#1A9B67", width=7)
    pill(draw, (1260, 665, 1555, 800), "控制与安全", fill="#ECF9F4", outline="#42A982", text_fill="#147A58", size=31)
    robot_arm(draw, 1700, 570, .45)


def visual_1_6(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    rounded(draw, (105, 350, 490, 795), fill="#F8FBFF")
    draw.text((298, 400), "当前场景", font=font(34, True), fill=NAVY, anchor="mm")
    robot_arm(draw, 250, 510, .55)
    draw.rectangle((180, 720, 420, 740), fill="#CAD4DE")
    draw.ellipse((350, 650, 405, 705), fill=YELLOW, outline="#C79400")
    arrow(draw, (510, 575), (640, 575), width=7)
    draw.text((575, 525), "预演", font=font(28, True), fill=BLUE, anchor="mm")
    branch_labels = spec.labels[1:4]
    outcomes = ("碰撞风险", "遮挡目标", "成功概率高")
    for i, (label, outcome) in enumerate(zip(branch_labels, outcomes)):
        x = 655 + i * 360
        rounded(draw, (x, 320, x + 320, 800), fill="#F7FBFF", outline="#87B4E7")
        pill(draw, (x + 25, 345, x + 295, 415), label, fill="#EAF5FF")
        robot_arm(draw, x + 115, 505, .45)
        color = "#1A9B67" if i == 2 else "#D95B5B"
        draw.ellipse((x + 115, 690, x + 205, 780), fill="#EFF8F5" if i == 2 else "#FFF0F0", outline=color, width=4)
        draw.text((x + 160, 735), "✓" if i == 2 else "×", font=font(48, True), fill=color, anchor="mm")
        draw.text((x + 160, 650), outcome, font=font(25, True), fill=color, anchor="mm")
    pill(draw, (1465, 455, 1795, 680), "选择更稳妥\n的动作", fill="#ECF9F4", outline="#1A9B67", text_fill="#147A58", size=33)


def visual_1_8(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    scenes = spec.labels[:5]
    for i, label in enumerate(scenes):
        x1 = 100 + i * 345
        y2, h = 815, 150 + i * 90
        draw.polygon([(x1, y2), (x1 + 290, y2), (x1 + 290, y2 - h), (x1, y2 - h + 55)], fill="#EAF5FF", outline=BLUE)
        draw.text((x1 + 145, y2 - h + 85), str(i + 1), font=font(38, True), fill=BLUE, anchor="mm")
        fit_text(draw, label, (x1 + 20, y2 - h + 120, x1 + 270, y2 - 18), 29, 20, True)
    arrow(draw, (140, 855), (1765, 855), color="#1A9B67", width=7)
    draw.text((950, 900), "开放性 · 任务时长 · 安全要求  持续上升", font=font(29, True), fill="#147A58", anchor="mm")


def visual_2_1(draw: ImageDraw.ImageDraw, spec: FigureSpec, image: Image.Image,
               asset_dir: Path) -> None:
    asset_path = asset_dir / "robot-arm.png"
    if asset_path.exists():
        with Image.open(asset_path) as source:
            source = source.convert("RGB").crop((280, 10, 1450, 930))
            source.thumbnail((690, 535), Image.Resampling.LANCZOS)
            image.paste(source, (960 - source.width // 2, 545 - source.height // 2))
    else:
        robot_arm(draw, 890, 470, 1.05)
    components = [(300, 360, "本体与机构", "机"), (300, 620, "传感器", "camera"), (610, 815, "执行器", "动"),
                  (1310, 815, "计算平台", "compute"), (1615, 620, "通信与同步", "联"), (1615, 360, "电源与安全", "shield")]
    for x, y, label, icon in components:
        sensor_icon(draw, x, y, icon)
        pill(draw, (x - 120, y + 70, x + 120, y + 135), label, size=24)
        arrow(draw, (x + (90 if x < 960 else -90), y), (850 if x < 960 else 1070, 590), color="#8AB5E9", width=4)


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
        fit_text(draw, name, (x1 + 28, top + 20, x1 + 223, bottom - 20), 28, 20, True, WHITE)
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
        draw.text((1478, y + 27), str(i + 1), font=font(18, True), fill=NAVY, anchor="mm")
        draw.text((1510, y + 27), item, font=font(25, True), fill=TEXT, anchor="lm")


def visual_2_4(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    mechanisms = spec.labels[:4]
    colors = ("#EAF5FF", "#EDF9F4", "#FFF7DD", "#F4EEFF")
    for i, (label, fill) in enumerate(zip(mechanisms, colors)):
        x = 100 + i * 435
        rounded(draw, (x, 315, x + 380, 555), fill=fill, outline="#78A9DF")
        draw.ellipse((x + 135, 340, x + 245, 450), fill=WHITE, outline=BLUE, width=4)
        draw.text((x + 190, 395), str(i + 1), font=font(39, True), fill=BLUE, anchor="mm")
        fit_text(draw, label, (x + 25, 465, x + 355, 535), 26, 19, True)
    rounded(draw, (180, 625, 1740, 845), fill="#FFF7F7", outline="#DB7777")
    draw.text((400, 680), "通信连通 ≠ 系统正确", font=font(36, True), fill="#B33F3F", anchor="mm")
    arrow(draw, (630, 730), (760, 730), color="#D95B5B", width=7)
    checks = spec.labels[5:]
    for i, item in enumerate(checks):
        x = 790 + i * 220
        pill(draw, (x, 675, x + 185, 790), item, fill=WHITE, outline="#D8A0A0", text_fill="#8B3D3D", size=26)


def generic_a(draw: ImageDraw.ImageDraw, spec: FigureSpec) -> None:
    labels = spec.labels
    count = len(labels)
    cols = min(4, count)
    rows = (count + cols - 1) // cols
    card_w, card_h = 330, 190
    gap_x, gap_y = 62, 52
    total_w = cols * card_w + (cols - 1) * gap_x
    left = (W - total_w) // 2
    top = 292 if rows == 2 else 390
    for index, label in enumerate(labels):
        row, col = divmod(index, cols)
        x = left + col * (card_w + gap_x)
        y = top + row * (card_h + gap_y)
        rounded(draw, (x, y, x + card_w, y + card_h), fill="#F7FBFF", outline="#75ADEB")
        draw.ellipse((x + 125, y + 24, x + 205, y + 104), fill=CYAN, outline=BLUE, width=4)
        draw.text((x + 165, y + 64), str(index + 1), font=font(34, True), fill=BLUE, anchor="mm")
        fit_text(draw, label, (x + 20, y + 112, x + card_w - 20, y + card_h - 12), 31, 22, True)


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
        fit_text(draw, label, (x + 92, y + 25, x + card_w - 18, y + card_h - 20), 33, 22, True)
        if col < cols - 1 and index + 1 < count:
            arrow(draw, (x + card_w + 9, y + card_h // 2), (x + card_w + gap - 9, y + card_h // 2), width=6)
        elif row < rows - 1 and index + 1 < count:
            arrow(draw, (x + card_w // 2, y + card_h + 8), (x + card_w // 2, y + card_h + 43), width=6)
    if spec.labels:
        fit_text(draw, " · ".join(spec.labels), (260, 852, 1660, 914), 25, 20, False, MUTED)


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
            fit_text(draw, label, (x + 45, y + 4, x + 475, y + 54), 25, 18, row == 0)


def render_figure(spec: FigureSpec, asset_dir: Path, output: Path) -> None:
    image = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(image)
    header(draw, spec)
    custom = {
        "1-1": visual_1_1,
        "1-2": visual_1_2,
        "1-4": visual_1_4,
        "1-5": visual_1_5,
        "1-6": visual_1_6,
        "1-8": visual_1_8,
        "2-1": visual_2_1,
        "2-2": visual_2_2,
        "2-4": visual_2_4,
    }.get(spec.key)
    if spec.key == "2-1":
        visual_2_1(draw, spec, image, asset_dir)
    elif custom:
        custom(draw, spec)
    elif spec.template == "A":
        generic_a(draw, spec)
    elif spec.template == "B":
        generic_b(draw, spec)
    else:
        generic_c(draw, spec)
    takeaway(draw, spec.takeaway)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True, icc_profile=SRGB_PROFILE)
