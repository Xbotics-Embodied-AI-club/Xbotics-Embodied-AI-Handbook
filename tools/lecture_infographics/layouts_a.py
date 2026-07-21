from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .assets import ASSETS, load_asset
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
SOFT = THEME.primary_soft
TEXT = THEME.text
MUTED = THEME.muted
YELLOW = THEME.accent
GREEN = THEME.success
RED = THEME.danger


def _asset_dir(asset_dir: Path) -> Path:
    if all((asset_dir / spec.filename).exists() for spec in ASSETS.values()):
        return asset_dir
    return Path(__file__).with_name("generated_assets")


def _paste_asset(
    image: Image.Image,
    asset_dir: Path,
    name: str,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    asset = load_asset(_asset_dir(asset_dir), name, (x2 - x1, y2 - y1))
    image.paste(asset, (x1 + (x2 - x1 - asset.width) // 2, y1 + (y2 - y1 - asset.height) // 2))


def _pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str = WHITE,
    outline: str = THEME.border,
    text_fill: str = TEXT,
    size: int = 28,
    weight: str = "bold",
) -> None:
    draw_card(draw, box, fill=fill, outline=outline, radius=17)
    fit_text(draw, text, box, size, 26, weight, text_fill)


def _badge(draw: ImageDraw.ImageDraw, center: tuple[int, int], number: int) -> None:
    x, y = center
    draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=BLUE)
    draw.text((x, y), str(number), font=font(26, "bold"), fill=WHITE, anchor="mm")


def _semantic_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], kind: str) -> None:
    x, y = center
    stroke = 4
    if kind == "body":
        draw.ellipse((x - 22, y - 36, x + 22, y + 8), outline=BLUE, width=stroke)
        draw.line((x - 34, y + 32, x, y + 8, x + 34, y + 32), fill=BLUE, width=stroke)
    elif kind == "sense":
        draw.rounded_rectangle((x - 38, y - 24, x + 38, y + 24), radius=9, outline=BLUE, width=stroke)
        draw.ellipse((x - 17, y - 17, x + 17, y + 17), outline=BLUE, width=stroke)
    elif kind == "state":
        draw.line((x - 38, y + 22, x - 18, y - 8, x, y + 6, x + 32, y - 30), fill=BLUE, width=stroke)
        draw.ellipse((x + 26, y - 36, x + 38, y - 24), fill=YELLOW)
    elif kind == "task":
        draw.rectangle((x - 31, y - 32, x + 31, y + 32), outline=BLUE, width=stroke)
        draw.line((x - 17, y - 10, x - 7, y, x + 16, y - 19), fill=GREEN, width=stroke)
        draw.line((x - 17, y + 16, x + 16, y + 16), fill=BLUE, width=stroke)
    elif kind == "policy":
        draw.ellipse((x - 34, y - 32, x + 34, y + 32), outline=BLUE, width=stroke)
        draw.line((x - 20, y + 18, x, y - 14, x + 20, y + 18), fill=BLUE, width=stroke)
        draw.ellipse((x - 5, y - 19, x + 5, y - 9), fill=YELLOW)
    elif kind == "control":
        draw.ellipse((x - 31, y - 31, x + 31, y + 31), outline=BLUE, width=stroke)
        draw.line((x, y, x + 20, y - 18), fill=BLUE, width=6)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=YELLOW)
    elif kind == "data":
        draw.ellipse((x - 36, y - 30, x + 36, y - 6), fill=SOFT, outline=BLUE, width=stroke)
        draw.rectangle((x - 36, y - 18, x + 36, y + 26), fill=SOFT, outline=BLUE, width=stroke)
        draw.arc((x - 36, y + 11, x + 36, y + 35), 0, 180, fill=BLUE, width=stroke)
    elif kind == "shield":
        draw.polygon(((x, y - 38), (x + 33, y - 24), (x + 27, y + 20), (x, y + 39), (x - 27, y + 20), (x - 33, y - 24)), fill="#ECF9F4", outline=GREEN)
        draw.line((x - 16, y, x - 4, y + 13, x + 20, y - 17), fill=GREEN, width=6)
    elif kind == "compute":
        draw.rounded_rectangle((x - 34, y - 30, x + 34, y + 30), radius=7, fill=SOFT, outline=BLUE, width=stroke)
        draw.rectangle((x - 15, y - 13, x + 15, y + 13), fill=BLUE)
        for offset in (-20, 0, 20):
            draw.line((x - 47, y + offset, x - 34, y + offset), fill=BLUE, width=3)
            draw.line((x + 34, y + offset, x + 47, y + offset), fill=BLUE, width=3)
    elif kind == "sensor":
        draw.rounded_rectangle((x - 39, y - 25, x + 39, y + 25), radius=10, fill=NAVY)
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill="#6CD4FF", outline=WHITE, width=4)
    elif kind == "actuator":
        draw.ellipse((x - 34, y - 34, x + 34, y + 34), fill=SOFT, outline=BLUE, width=stroke)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=BLUE)
        for dx, dy in ((0, -45), (0, 45), (-45, 0), (45, 0)):
            draw.line((x, y, x + dx, y + dy), fill=BLUE, width=5)
    elif kind == "network":
        for dx, dy in ((0, -30), (-34, 24), (34, 24)):
            draw.ellipse((x + dx - 11, y + dy - 11, x + dx + 11, y + dy + 11), fill=BLUE)
        draw.line((x, y - 19, x - 25, y + 15, x + 25, y + 15), fill=BLUE, width=stroke)


def draw_1_1(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw_card(draw, (90, 260, 785, 890), fill="#F6F8FB", outline="#B8C9DB")
    draw_card(draw, (1135, 260, 1830, 890), fill="#F1F8FF", outline=BLUE)
    draw.text((438, 312), "数字 AI", font=font(36, "bold"), fill=NAVY, anchor="mm")
    draw.text((1482, 312), "具身智能", font=font(36, "bold"), fill=NAVY, anchor="mm")
    draw.rounded_rectangle((220, 375, 655, 650), radius=22, fill=NAVY)
    draw.rectangle((242, 398, 633, 602), fill=WHITE)
    for index, width in enumerate((275, 225, 298, 175)):
        draw.rounded_rectangle((285, 445 + index * 36, 285 + width, 459 + index * 36), radius=7, fill="#7F9DBD")
    draw.rounded_rectangle((360, 664, 516, 686), radius=10, fill="#8DA2B9")
    _pill(draw, (260, 738, 615, 820), "文本 / 图像", fill=WHITE, outline="#9CB4CB", text_fill=MUTED, size=30)
    _paste_asset(image, asset_dir, "desktop_pick", (1170, 350, 1795, 775))
    _pill(draw, (1245, 787, 1720, 855), "感知环境并执行动作", fill=WHITE, outline=BLUE, size=28)
    draw_arrow(draw, (810, 460), (1110, 460), width=8)
    draw.text((960, 415), "行动", font=font(30, "bold"), fill=BLUE, anchor="mm")
    draw_arrow(draw, (1110, 710), (810, 710), semantic="success", width=7)
    draw.text((960, 755), "反馈", font=font(30, "bold"), fill=GREEN, anchor="mm")


def draw_1_2(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw.ellipse((690, 300, 1230, 840), fill="#F3F9FF", outline="#78AFE9", width=5)
    _paste_asset(image, asset_dir, "robot_arm_camera", (720, 330, 1200, 750))
    _pill(draw, (790, 770, 1130, 844), "具身智能体", fill=BLUE, outline=BLUE, text_fill=WHITE, size=32)
    items = [
        ((165, 325, 485, 445), "身体", "body", (650, 430)),
        ((145, 510, 465, 630), "感知", "sense", (680, 550)),
        ((205, 695, 525, 815), "状态", "state", (715, 675)),
        ((1435, 325, 1755, 445), "任务", "task", (1270, 430)),
        ((1455, 510, 1775, 630), "策略", "policy", (1240, 550)),
        ((1395, 695, 1715, 815), "控制", "control", (1205, 675)),
        ((800, 250, 1120, 338), "数据", "data", (960, 350)),
    ]
    for index, (box, label, kind, endpoint) in enumerate(items, start=1):
        x1, y1, x2, y2 = box
        start = ((x1 + x2) // 2, (y1 + y2) // 2)
        draw.line((*start, *endpoint), fill="#8CB9EC", width=4)
        draw_card(draw, box, fill=WHITE, outline="#8CB9EC", radius=20)
        _semantic_icon(draw, (x1 + 68, (y1 + y2) // 2), kind)
        fit_text(draw, label, (x1 + 115, y1 + 12, x2 - 18, y2 - 12), 30, 26, "bold", NAVY)
        _badge(draw, (x1 + 25, y1 + 25), index)


def draw_1_4(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    _paste_asset(image, asset_dir, "robot_arm", (90, 370, 600, 860))
    _pill(draw, (120, 280, 570, 345), "共同服务于可靠行动", fill=SOFT, outline=BLUE, size=28)
    stages = [
        (560, 688, "传统 Pipeline", "可靠执行", "#EAF4FF"),
        (790, 606, "模仿学习", "从示范学习", "#EEF8F4"),
        (1020, 524, "强化学习", "探索优化", "#FFF7DD"),
        (1250, 442, "VLA", "多模态泛化", "#F2EEFF"),
        (1480, 360, "世界模型", "预测未来", "#FFF0EA"),
    ]
    for index, (x, y, title, role, fill) in enumerate(stages, start=1):
        draw_card(draw, (x, y, x + 245, 868), fill=fill, outline="#78ABE3", radius=22)
        _badge(draw, (x + 122, y + 32), index)
        fit_text(draw, title, (x + 12, y + 60, x + 233, y + 108), 28, 26, "bold", NAVY)
        draw.line((x + 24, y + 112, x + 221, y + 112), fill="#B6CDE7", width=3)
        fit_text(draw, role, (x + 18, y + 118, x + 227, min(y + 170, 856)), 27, 26, "regular", TEXT)
    draw.text((1210, 295), "能力扩展方向  ↗", font=font(28, "bold"), fill=BLUE, anchor="mm")


def draw_1_5(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    inputs = (("视觉", "sense", 300), ("语言", "task", 510), ("机器人状态", "state", 720))
    for label, kind, y in inputs:
        draw_card(draw, (120, y, 500, y + 130), fill=WHITE, outline="#86B4E8", radius=22)
        _semantic_icon(draw, (195, y + 65), kind)
        fit_text(draw, label, (260, y + 18, 480, y + 112), 30, 26, "bold", NAVY)
        draw_arrow(draw, (510, y + 65), (710, 570), semantic="secondary", width=4)
    draw_card(draw, (720, 340, 1160, 800), fill="#EEF6FF", outline=BLUE, width=4, radius=32)
    draw.ellipse((825, 390, 1055, 620), fill=WHITE, outline=BLUE, width=5)
    draw.text((940, 478), "VLA", font=font(54, "bold"), fill=BLUE, anchor="mm")
    draw.text((940, 548), "多模态融合", font=font(30, "bold"), fill=NAVY, anchor="mm")
    _pill(draw, (790, 660, 1090, 742), "生成动作建议", fill="#FFF7DD", outline="#D9AA25", text_fill="#896300", size=28)
    draw_arrow(draw, (1170, 570), (1295, 570), width=8)
    _paste_asset(image, asset_dir, "robot_arm_camera", (1295, 275, 1790, 700))
    _pill(draw, (1335, 712, 1750, 785), "动作建议", fill="#FFF7DD", outline="#D9AA25", text_fill="#896300", size=30)
    draw_arrow(draw, (1542, 795), (1542, 832), semantic="success", width=7)
    _pill(draw, (1305, 837, 1778, 900), "控制与安全负责最终执行", fill="#ECF9F4", outline=GREEN, text_fill=GREEN, size=27)


def draw_1_6(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw_card(draw, (90, 285, 515, 870), fill="#F7FAFD", outline="#9CB8D4", radius=24)
    draw.text((302, 330), "当前场景", font=font(32, "bold"), fill=NAVY, anchor="mm")
    _paste_asset(image, asset_dir, "desktop_pick", (115, 375, 490, 685))
    _pill(draw, (150, 742, 455, 824), "生成候选动作", fill=SOFT, outline=BLUE, size=28)
    draw_arrow(draw, (525, 575), (630, 575), width=8)
    draw.text((578, 527), "预演", font=font(28, "bold"), fill=BLUE, anchor="mm")
    cards = [
        (640, "左侧抓取", "碰撞风险", RED, "danger"),
        (1015, "上方抓取", "遮挡目标", RED, "danger"),
        (1390, "先移开障碍", "成功概率高", GREEN, "success"),
    ]
    for index, (x, action, outcome, color, style) in enumerate(cards):
        draw_card(draw, (x, 295, x + 330, 870), fill=WHITE, outline="#8AB5E7", radius=24)
        _pill(draw, (x + 28, 325, x + 302, 395), action, fill=SOFT, outline="#8AB5E7", size=27)
        _paste_asset(image, asset_dir, "robot_arm", (x + 35, 420, x + 295, 660))
        draw_card(draw, (x + 35, 690, x + 295, 825), style=style, radius=18)
        symbol = "✓" if color == GREEN else "×"
        draw.text((x + 80, 758), symbol, font=font(46, "bold"), fill=color, anchor="mm")
        fit_text(draw, outcome, (x + 122, 708, x + 282, 808), 27, 26, "bold", color)
        if index == 2:
            draw.rounded_rectangle((x + 15, 280, x + 315, 883), radius=28, outline=GREEN, width=6)
            draw.text((x + 165, 842), "推荐", font=font(26, "bold"), fill=GREEN, anchor="mm")


def draw_1_8(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    scenes = [
        ("桌面抓取", "desktop_pick", 610),
        ("移动操作", "mobile_manipulator", 540),
        ("工业 / 商超", "robot_arm", 470),
        ("家庭服务", "mobile_manipulator", 400),
        ("人形机器人", "humanoid_robot", 330),
    ]
    for index, (label, asset_name, top) in enumerate(scenes):
        x = 90 + index * 354
        bottom = 860
        draw_card(draw, (x, top, x + 312, bottom), fill="#F5FAFF", outline="#7CACDF", radius=22)
        _badge(draw, (x + 38, top + 38), index + 1)
        _paste_asset(image, asset_dir, asset_name, (x + 18, top + 55, x + 294, bottom - 78))
        _pill(draw, (x + 18, bottom - 78, x + 294, bottom - 15), label, fill=WHITE, outline="#91B9E5", size=27)
        if index < len(scenes) - 1:
            draw_arrow(draw, (x + 312, top + 95), (x + 344, top + 68), semantic="secondary", width=4)
    draw_arrow(draw, (145, 882), (1765, 882), semantic="accent", width=7)
    draw.text((960, 915), "开放性  ·  任务时长  ·  安全要求  持续上升", font=font(28, "bold"), fill="#8A6500", anchor="mm")


def draw_2_1(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw.ellipse((620, 275, 1300, 895), fill="#F2F8FF", outline="#7AAFE8", width=5)
    _paste_asset(image, asset_dir, "robot_arm_camera", (650, 300, 1270, 835))
    _pill(draw, (785, 814, 1135, 880), "机器人硬件系统", fill=BLUE, outline=BLUE, text_fill=WHITE, size=30)
    items = [
        ((105, 290, 480, 420), "本体与机构", "body", (665, 410)),
        ((85, 500, 460, 630), "传感器", "sensor", (630, 555)),
        ((155, 710, 530, 840), "执行器", "actuator", (680, 720)),
        ((1440, 290, 1815, 420), "计算平台", "compute", (1255, 410)),
        ((1460, 500, 1835, 630), "通信与同步", "network", (1290, 555)),
        ((1390, 710, 1765, 840), "电源与安全", "shield", (1240, 720)),
    ]
    for index, (box, label, kind, endpoint) in enumerate(items, start=1):
        x1, y1, x2, y2 = box
        start = ((x1 + x2) // 2, (y1 + y2) // 2)
        draw.line((*start, *endpoint), fill="#80AFE3", width=4)
        draw_card(draw, box, fill=WHITE, outline="#80AFE3", radius=22)
        _semantic_icon(draw, (x1 + 78, (y1 + y2) // 2), kind)
        fit_text(draw, label, (x1 + 140, y1 + 18, x2 - 18, y2 - 18), 29, 26, "bold", NAVY)
        _badge(draw, (x1 + 27, y1 + 27), index)


A_LAYOUTS = {
    "1-1": draw_1_1,
    "1-2": draw_1_2,
    "1-4": draw_1_4,
    "1-5": draw_1_5,
    "1-6": draw_1_6,
    "1-8": draw_1_8,
    "2-1": draw_2_1,
}


def render_a(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw_header(draw, spec)
    A_LAYOUTS[spec.key](image, draw, spec, asset_dir)
    draw_takeaway(draw, spec.takeaway)
