from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .components import draw_arrow, draw_card, draw_header, draw_takeaway, fit_text, font
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

LOOP_FIGURES = {"1-3", "1-7", "1-9", "2-3", "2-5"}


def _path_arrow(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[int, int], ...],
    *,
    semantic: str = "primary",
    width: int = 7,
) -> None:
    colors = {
        "primary": BLUE,
        "secondary": THEME.border,
        "success": GREEN,
        "danger": RED,
        "accent": YELLOW,
    }
    color = colors.get(semantic, semantic)
    if len(points) < 2:
        raise ValueError("an arrow path needs at least two points")
    if len(points) > 2:
        draw.line(points[:-1], fill=color, width=width, joint="curve")
    draw_arrow(draw, points[-2], points[-1], color=color, width=width)


def _icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], kind: str) -> None:
    x, y = center
    stroke = 4
    if kind == "target":
        draw.ellipse((x - 31, y - 31, x + 31, y + 31), outline=BLUE, width=stroke)
        draw.ellipse((x - 17, y - 17, x + 17, y + 17), outline=BLUE, width=stroke)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=RED)
    elif kind == "camera":
        draw.rounded_rectangle((x - 36, y - 24, x + 36, y + 24), radius=8, outline=BLUE, width=stroke)
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=BLUE, width=stroke)
        draw.rectangle((x - 25, y - 33, x - 3, y - 24), fill=BLUE)
    elif kind == "state":
        draw.line((x - 35, y + 22, x - 16, y - 5, x + 2, y + 9, x + 31, y - 28), fill=BLUE, width=stroke)
        for px, py in ((x - 35, y + 22), (x - 16, y - 5), (x + 2, y + 9), (x + 31, y - 28)):
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=YELLOW, outline=BLUE)
    elif kind == "policy":
        draw.ellipse((x - 32, y - 31, x + 32, y + 31), outline=BLUE, width=stroke)
        draw.line((x - 21, y + 18, x, y - 15, x + 21, y + 18), fill=BLUE, width=stroke)
        draw.ellipse((x - 6, y - 21, x + 6, y - 9), fill=YELLOW)
    elif kind == "control":
        draw.ellipse((x - 30, y - 30, x + 30, y + 30), outline=BLUE, width=stroke)
        draw.line((x, y, x + 22, y - 17), fill=BLUE, width=6)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=YELLOW)
    elif kind == "check":
        draw.polygon(((x, y - 34), (x + 31, y - 20), (x + 25, y + 20), (x, y + 36), (x - 25, y + 20), (x - 31, y - 20)), fill="#ECF9F4", outline=GREEN)
        draw.line((x - 15, y, x - 4, y + 12, x + 20, y - 16), fill=GREEN, width=6)
    elif kind == "data":
        draw.ellipse((x - 34, y - 28, x + 34, y - 7), fill=SOFT, outline=BLUE, width=stroke)
        draw.rectangle((x - 34, y - 18, x + 34, y + 24), fill=SOFT, outline=BLUE, width=stroke)
        draw.arc((x - 34, y + 10, x + 34, y + 31), 0, 180, fill=BLUE, width=stroke)
    elif kind == "filter":
        draw.polygon(((x - 35, y - 28), (x + 35, y - 28), (x + 10, y), (x + 10, y + 29), (x - 10, y + 29), (x - 10, y)), fill=SOFT, outline=BLUE)
    elif kind == "train":
        draw.rounded_rectangle((x - 34, y - 28, x + 34, y + 28), radius=8, outline=BLUE, width=stroke)
        draw.rectangle((x - 14, y - 11, x + 14, y + 11), fill=BLUE)
        for offset in (-19, 0, 19):
            draw.line((x - 45, y + offset, x - 34, y + offset), fill=BLUE, width=3)
            draw.line((x + 34, y + offset, x + 45, y + offset), fill=BLUE, width=3)
    elif kind == "evaluate":
        draw.line((x - 30, y + 27, x - 12, y + 3, x + 5, y + 15, x + 29, y - 24), fill=GREEN, width=5)
        draw.line((x - 31, y + 31, x + 34, y + 31), fill=BLUE, width=stroke)
        draw.line((x - 31, y + 31, x - 31, y - 27), fill=BLUE, width=stroke)
    elif kind == "deploy":
        draw.polygon(((x, y - 35), (x + 27, y + 20), (x + 8, y + 14), (x, y + 34), (x - 8, y + 14), (x - 27, y + 20)), fill=SOFT, outline=BLUE)
    elif kind == "error":
        draw.line((x - 32, y, x + 32, y), fill=BLUE, width=stroke)
        draw.line((x, y - 32, x, y + 32), fill=BLUE, width=stroke)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=WHITE, outline=RED, width=4)
    elif kind == "limit":
        draw.line((x - 31, y, x + 31, y), fill=BLUE, width=5)
        draw.polygon(((x - 31, y), (x - 16, y - 12), (x - 16, y + 12)), fill=YELLOW)
        draw.polygon(((x + 31, y), (x + 16, y - 12), (x + 16, y + 12)), fill=YELLOW)
        draw.line((x - 31, y - 29, x - 31, y + 29), fill=BLUE, width=stroke)
        draw.line((x + 31, y - 29, x + 31, y + 29), fill=BLUE, width=stroke)
    elif kind == "move":
        draw.line((x - 32, y + 23, x + 22, y - 25), fill=BLUE, width=6)
        draw.polygon(((x + 30, y - 32), (x + 20, y - 8), (x + 7, y - 23)), fill=BLUE)
        draw.ellipse((x - 38, y + 17, x - 26, y + 29), fill=YELLOW)
    elif kind == "motor":
        draw.ellipse((x - 30, y - 30, x + 30, y + 30), outline=BLUE, width=stroke)
        draw.text((x, y), "M", font=font(28, "bold"), fill=BLUE, anchor="mm")
    elif kind == "robot":
        draw.rounded_rectangle((x - 31, y - 25, x + 31, y + 25), radius=8, outline=BLUE, width=stroke)
        draw.ellipse((x - 17, y - 6, x - 7, y + 4), fill=BLUE)
        draw.ellipse((x + 7, y - 6, x + 17, y + 4), fill=BLUE)
        draw.line((x - 15, y + 13, x + 15, y + 13), fill=YELLOW, width=5)
        draw.line((x, y - 34, x, y - 25), fill=BLUE, width=stroke)
    elif kind == "sensor":
        draw.arc((x - 31, y - 31, x + 31, y + 31), 210, 330, fill=BLUE, width=stroke)
        draw.arc((x - 21, y - 21, x + 21, y + 21), 210, 330, fill=BLUE, width=stroke)
        draw.ellipse((x - 6, y + 12, x + 6, y + 24), fill=YELLOW, outline=BLUE)
    elif kind == "publish":
        draw.rectangle((x - 27, y - 31, x + 18, y + 31), fill=SOFT, outline=BLUE, width=stroke)
        draw.line((x - 14, y - 14, x + 5, y - 14), fill=BLUE, width=3)
        draw.line((x - 14, y, x + 5, y), fill=BLUE, width=3)
        draw.line((x - 14, y + 14, x + 5, y + 14), fill=BLUE, width=3)
        draw.polygon(((x + 12, y - 10), (x + 36, y), (x + 12, y + 10)), fill=YELLOW)
    elif kind == "status":
        draw.ellipse((x - 31, y - 31, x + 31, y + 31), outline=BLUE, width=stroke)
        draw.line((x - 16, y + 2, x - 3, y + 15, x + 20, y - 15), fill=GREEN, width=6)
    else:
        raise ValueError(f"unknown semantic icon: {kind}")


def _step_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    kind: str,
    *,
    fill: str = WHITE,
    outline: str = THEME.border,
    text_size: int = 28,
) -> None:
    x1, y1, x2, y2 = box
    draw_card(draw, box, fill=fill, outline=outline, radius=20)
    _icon(draw, (x1 + 62, (y1 + y2) // 2), kind)
    fit_text(draw, label, (x1 + 105, y1 + 12, x2 - 16, y2 - 12), text_size, 24, "bold", NAVY)


def _topic(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, *, color: str = BLUE) -> None:
    draw.rounded_rectangle(box, radius=15, fill=WHITE, outline=color, width=3)
    fit_text(draw, label, box, 27, 24, "bold", color)


def draw_1_3(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    cards = [
        ((90, 300, 390, 420), "任务输入", "target"),
        ((450, 300, 750, 420), "环境感知", "camera"),
        ((810, 300, 1110, 420), "状态估计", "state"),
        ((1170, 300, 1470, 420), "策略与规划", "policy"),
        ((1470, 550, 1770, 670), "控制执行", "control"),
        ((1110, 760, 1410, 880), "结果检查", "check"),
        ((630, 760, 930, 880), "数据记录", "data"),
    ]
    for box, label, kind in cards:
        _step_card(draw, box, label, kind)
    for start, end in (
        ((390, 360), (450, 360)), ((750, 360), (810, 360)), ((1110, 360), (1170, 360)),
        ((1470, 360), (1618, 550)), ((1618, 670), (1410, 820)), ((1110, 820), (930, 820)),
    ):
        draw_arrow(draw, start, end, width=7)
    _path_arrow(draw, ((630, 820), (250, 820), (250, 485), (600, 485), (600, 420)), semantic="success", width=7)
    draw.text((395, 466), "新反馈驱动下一次感知", font=font(24, "bold"), fill=GREEN, anchor="mm")
    draw.text((960, 605), "执行链", font=font(34, "bold"), fill=BLUE, anchor="mm")
    draw.text((960, 650), "边做 · 边看 · 边修正", font=font(28), fill=MUTED, anchor="mm")


def draw_1_7(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    cards = [
        ((100, 350, 390, 475), "真实执行", "robot"),
        ((455, 350, 745, 475), "记录数据", "data"),
        ((810, 350, 1100, 475), "筛选与标注", "filter"),
        ((1165, 350, 1455, 475), "模型训练", "train"),
        ((1165, 680, 1455, 805), "评测", "evaluate"),
        ((810, 680, 1100, 805), "重新部署", "deploy"),
    ]
    for box, label, kind in cards:
        _step_card(draw, box, label, kind)
    for start, end in (
        ((390, 412), (455, 412)), ((745, 412), (810, 412)), ((1100, 412), (1165, 412)),
        ((1310, 475), (1310, 680)), ((1165, 742), (1100, 742)),
    ):
        draw_arrow(draw, start, end, width=7)
    _path_arrow(draw, ((810, 742), (245, 742), (245, 475)), semantic="success", width=7)
    draw.text((455, 708), "新版系统回到真实世界", font=font(24, "bold"), fill=GREEN, anchor="mm")
    _path_arrow(draw, ((1455, 742), (1735, 742), (1735, 275), (955, 275), (955, 350)), semantic="danger", width=7)
    _topic(draw, (1470, 500, 1775, 570), "失败样本", color=RED)
    draw.text((1538, 305), "回到筛选 / 训练", font=font(24, "bold"), fill=RED, anchor="mm")


def _draw_xy_plane(draw: ImageDraw.ImageDraw) -> None:
    draw_card(draw, (700, 500, 1220, 715), fill="#F8FBFF", outline="#8CB9EC", radius=22)
    draw.line((760, 665, 1165, 665), fill=NAVY, width=4)
    draw.line((760, 665, 760, 535), fill=NAVY, width=4)
    draw.text((1175, 665), "x", font=font(24, "bold"), fill=NAVY, anchor="lm")
    draw.text((760, 523), "y", font=font(24, "bold"), fill=NAVY, anchor="mb")
    draw.ellipse((815, 620, 837, 642), fill=BLUE)
    draw.ellipse((1080, 555, 1106, 581), fill=RED)
    draw.line((837, 626, 965, 595), fill=GREEN, width=7)
    draw.polygon(((978, 592), (956, 582), (963, 605)), fill=GREEN)
    draw.text((826, 681), "当前位置", font=font(24, "bold"), fill=BLUE, anchor="mm")
    draw.text((1093, 540), "目标点", font=font(24, "bold"), fill=RED, anchor="mm")
    draw.text((960, 690), "受限制的单步移动", font=font(24), fill=GREEN, anchor="mm")


def draw_1_9(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    cards = [
        ((80, 300, 365, 415), "输入目标点", "target"),
        ((415, 300, 700, 415), "读取当前位置", "camera"),
        ((750, 300, 1035, 415), "计算误差", "error"),
        ((1085, 300, 1370, 415), "限制单步移动", "limit"),
        ((1415, 500, 1790, 615), "更新当前位置", "move"),
        ((1415, 730, 1790, 845), "判断是否到达", "check"),
        ((80, 730, 455, 845), "记录轨迹", "data"),
    ]
    for box, label, kind in cards:
        _step_card(draw, box, label, kind, text_size=27)
    for start, end in (
        ((365, 358), (415, 358)), ((700, 358), (750, 358)), ((1035, 358), (1085, 358)),
        ((1370, 358), (1602, 500)), ((1602, 615), (1602, 730)),
    ):
        draw_arrow(draw, start, end, width=7)
    draw_arrow(draw, (1415, 787), (455, 787), width=7)
    _path_arrow(draw, ((80, 787), (55, 787), (55, 465), (557, 465), (557, 415)), semantic="success", width=7)
    draw.text((305, 446), "未到达：用最新位置继续修正", font=font(24, "bold"), fill=GREEN, anchor="mm")
    _draw_xy_plane(draw)
    _topic(draw, (190, 540, 585, 610), spec.labels[0], color=MUTED)


def draw_2_3(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    _topic(draw, (90, 275, 350, 340), "策略闭环", color=BLUE)
    _topic(draw, (90, 495, 350, 560), "机器人控制闭环", color="#B07A00")
    _topic(draw, (90, 715, 350, 780), "电机局部闭环", color=GREEN)
    cards = [
        ((430, 270, 720, 385), "任务目标", "target"),
        ((805, 270, 1095, 385), "规划 / 策略", "policy"),
        ((1180, 470, 1470, 585), "控制与安全", "check"),
        ((1180, 690, 1470, 805), "驱动命令", "control"),
        ((805, 790, 1095, 905), "执行器与本体", "motor"),
        ((430, 690, 720, 805), "传感器反馈", "sensor"),
        ((430, 470, 720, 585), "驱动与\n状态估计", "state"),
    ]
    for box, label, kind in cards:
        _step_card(draw, box, label, kind, text_size=27)
    for start, end in (
        ((720, 327), (805, 327)), ((1095, 327), (1325, 470)), ((1325, 585), (1325, 690)),
        ((1180, 747), (1095, 847)), ((805, 847), (720, 747)), ((575, 690), (575, 585)),
    ):
        draw_arrow(draw, start, end, width=7)
    _path_arrow(draw, ((430, 527), (380, 527), (380, 420), (950, 420), (950, 385)), semantic="success", width=7)
    draw.text((700, 401), "状态上行", font=font(24, "bold"), fill=GREEN, anchor="mm")
    draw.text((1512, 598), "指令下行", font=font(24, "bold"), fill=BLUE, anchor="lm")
    draw.text((1700, 330), "慢", font=font(30, "bold"), fill=BLUE, anchor="mm")
    draw_arrow(draw, (1700, 380), (1700, 820), semantic="accent", width=7)
    draw.text((1700, 870), "快", font=font(30, "bold"), fill="#8A6500", anchor="mm")


def _node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, kind: str, *, outline: str = THEME.border) -> None:
    x1, y1, x2, y2 = box
    draw_card(draw, box, fill=WHITE, outline=outline, radius=22)
    _icon(draw, ((x1 + x2) // 2, y1 + 58), kind)
    fit_text(draw, label, (x1 + 15, y1 + 106, x2 - 15, y2 - 14), 27, 24, "bold", NAVY)


def draw_2_5(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    nodes = [
        ((90, 285, 390, 455), "target_publisher", "publish"),
        ((90, 650, 390, 820), "robot_state_node", "state"),
        ((620, 345, 920, 515), "policy_node", "policy"),
        ((1200, 345, 1500, 515), "controller_node", "control"),
        ((620, 700, 920, 870), "task_status_node", "status"),
        ((1460, 700, 1760, 870), "episode_recorder", "data"),
    ]
    for box, label, kind in nodes:
        _node(draw, box, label, kind, outline=BLUE if "policy" in label or "controller" in label else THEME.border)

    draw_arrow(draw, (390, 370), (620, 415), width=7)
    _topic(draw, (405, 300, 610, 360), "/target_joint")

    draw_arrow(draw, (920, 430), (1200, 430), width=7)
    _topic(draw, (930, 345, 1190, 405), "/action_command")

    _path_arrow(draw, ((1200, 485), (1080, 485), (1080, 605), (390, 735)), semantic="success", width=7)
    draw.text((825, 595), "执行后更新机器人状态", font=font(24, "bold"), fill=GREEN, anchor="mm")

    _path_arrow(draw, ((390, 690), (505, 690), (505, 490), (620, 490)), semantic="success", width=7)
    _topic(draw, (405, 615, 605, 675), "/joint_states", color=GREEN)

    draw_arrow(draw, (390, 755), (620, 785), semantic="success", width=7)
    _topic(draw, (405, 785, 605, 845), "/joint_states", color=GREEN)

    draw_arrow(draw, (920, 785), (1460, 785), width=7)
    _topic(draw, (1085, 700, 1295, 760), "/task_status")

    draw.text((960, 920), "记录状态与结果，支撑下一轮迭代", font=font(24), fill=MUTED, anchor="mm")


B_LAYOUTS = {
    "1-3": draw_1_3,
    "1-7": draw_1_7,
    "1-9": draw_1_9,
    "2-3": draw_2_3,
    "2-5": draw_2_5,
}


def render_b(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None:
    draw_header(draw, spec)
    B_LAYOUTS[spec.key](image, draw, spec, asset_dir)
    draw_takeaway(draw, spec.takeaway)
