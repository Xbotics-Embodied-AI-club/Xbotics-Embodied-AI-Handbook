from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    primary: str = "#0B5FCC"
    primary_dark: str = "#12335B"
    primary_soft: str = "#EAF4FF"
    accent: str = "#FFC83D"
    success: str = "#159A6E"
    danger: str = "#D95858"
    text: str = "#24364B"
    muted: str = "#64758B"
    border: str = "#A9CBEF"
    canvas: str = "#FFFFFF"
    title_size: int = 66
    subtitle_size: int = 30
    section_size: int = 34
    body_size: int = 28
    small_size: int = 24
    min_text_size: int = 24
    card_radius: int = 24
    card_border: int = 3
    primary_arrow: int = 8
    secondary_arrow: int = 4


THEME = Theme()
