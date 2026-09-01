"""统一的出图字体规范：西文/数字用 Times New Roman，中文用宋体。

用合并好的 TimesSong.ttf —— 一套字体里西文是 Times New Roman、中文是宋体，所以同一段
中英文混排的文本也能正确显示。**必须用合并字体，不能配两个字体名**：matplotlib 一段文本
只用一种字体、不逐字回退，中英混排的坐标轴标签配两个字体名，结果是其中一种语言出问题。

字体二进制不在本仓库里。TimesSong 合并的 Times New Roman 与宋体都是专有字体，本仓是公开仓，
把它提交进来就是再分发。所以字体路径由环境变量 `XBOTICS_FIG_FONT` 给出，**取不到就报错停下**：
独立克隆本仓的人跑出图脚本会看到一句说明并中止，**这是正确行为，不是 bug** ——
悄悄回落到系统字体只会渲出一批字体不对却看不出来的图。

所有出图脚本 import 本模块并调用 apply() 即可。
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ENV_VAR = "XBOTICS_FIG_FONT"

_MISSING = f"""出图字体没配好，已中止（没有回落到系统字体，那样会渲出一批看不出错的错图）。

本仓是公开仓，不收专有字体二进制，所以字体路径走环境变量 {ENV_VAR}，指向合并字体
TimesSong.ttf（西文 Times New Roman + 中文宋体）。

  · 在父仓 Xbotics2 里开发：`direnv allow` 即可，父仓 .envrc 已经声明了它。
  · 独立克隆本仓：自备一份合并字体，然后
        export {ENV_VAR}=/path/to/TimesSong.ttf
"""


def font_path() -> Path:
    """拿到字体文件路径；没配或者路径不存在都直接中止。"""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        raise SystemExit(f"{_MISSING}\n当前 {ENV_VAR} 未设置。")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise SystemExit(f"{_MISSING}\n当前 {ENV_VAR}={raw} —— 这个路径上没有文件。")
    return path


def assert_covered(text: str, where: str = "") -> None:
    """字体缺哪个字就报错，别让它悄悄渲成豆腐块。

    matplotlib 遇到字体没有的字符只发一条 warning、照样出图，产物里是一个空方框——
    正是本仓最忌讳的那种「跑完了、看着成功了、其实是错的」。所以这里直接中止。

    实际踩过：TimesSong 没有 ✓(U+2713)、∇(U+2207)、−(U+2212)，
    而 20902 个 CJK 汉字它都有——只查中文覆盖会得出「一个都不缺」的错误结论。
    """
    from fontTools.ttLib import TTFont

    path = font_path()
    covered = set()
    for table in TTFont(path)["cmap"].tables:
        covered |= set(table.cmap)
    missing = sorted({c for c in text if ord(c) > 127 and ord(c) not in covered})
    if missing:
        detail = "、".join(f"{c!r}(U+{ord(c):04X})" for c in missing)
        raise SystemExit(
            f"字体缺字，已中止{('：' + where) if where else ''}——{detail}。\n"
            f"字体：{path}\n"
            f"这些字符会渲成空方框。改用字体里有的字符，或把它画出来"
            f"（本仓已有先例：对勾用线段画，不用 ✓ 字形）。")


def apply() -> str:
    """把字体装进 matplotlib 并设成全局默认。

    Returns:
        字体族名（如 "TimesSong"）。调用方把它写进图片元数据，
        用图本身证明这一张确实是用这套字体渲的。
    """
    path = font_path()
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"      # 公式走 Times 风格
    return name


if __name__ == "__main__":
    print(f"字体已生效：{apply()}  ←  {font_path()}")
