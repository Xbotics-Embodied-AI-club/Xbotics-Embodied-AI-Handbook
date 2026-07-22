from __future__ import annotations

from pathlib import Path

from .manifest import FIGURES
from .render import render_figure


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = {
    1: ROOT / "docs/part1-system-basics/part1_picture/Lecture1_picture",
    2: ROOT / "docs/part1-system-basics/part1_picture/Lecture2_picture",
}


def main() -> None:
    asset_dir = Path(__file__).with_name("generated_assets")
    for spec in FIGURES.values():
        output = OUTPUTS[spec.lecture] / f"figure-{spec.key}.png"
        render_figure(spec, asset_dir, output)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
