from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


EXPECTED_KEYS = {
    *(f"1-{number}" for number in range(1, 10)),
    *(f"2-{number}" for number in range(1, 6)),
}


def load_manifest():
    try:
        from tools.lecture_infographics.manifest import FIGURES
    except ImportError as exc:
        raise AssertionError("manifest module must exist") from exc
    return FIGURES


class ManifestTests(unittest.TestCase):
    def test_manifest_has_exactly_fourteen_expected_figures(self):
        figures = load_manifest()
        self.assertEqual(set(figures), EXPECTED_KEYS)
        self.assertEqual(len(figures), 14)

    def test_figure_metadata_matches_template_rules(self):
        figures = load_manifest()
        for key, spec in figures.items():
            self.assertIn(spec.template, {"A", "B", "C"}, key)
            self.assertTrue(spec.title.strip(), key)
            self.assertTrue(spec.takeaway.strip(), key)
            self.assertEqual(spec.key, key)
            self.assertEqual(f"{spec.lecture}-{spec.number}", key)
            if spec.lecture == 1:
                self.assertNotEqual(spec.template, "C", key)
            if spec.template == "B":
                self.assertGreaterEqual(len(spec.steps), 2, key)

    def test_required_exact_labels_are_present(self):
        figures = load_manifest()
        self.assertEqual(
            figures["1-2"].labels,
            ("身体", "感知", "状态", "任务", "策略", "控制", "数据"),
        )
        self.assertIn("二维闭环示意，不是机械臂仿真", figures["1-9"].labels)
        self.assertIn("ROS2 / 消息接口", figures["2-2"].labels)
        self.assertEqual(len(figures["2-5"].steps), 6)


class ThemeTests(unittest.TestCase):
    def test_typography_floor_and_hierarchy(self):
        from tools.lecture_infographics.theme import THEME
        self.assertGreaterEqual(THEME.title_size, 64)
        self.assertLessEqual(THEME.title_size, 68)
        self.assertEqual(THEME.subtitle_size, 30)
        self.assertGreaterEqual(THEME.body_size, 26)
        self.assertGreaterEqual(THEME.min_text_size, 24)
        self.assertGreater(THEME.title_size, THEME.section_size)
        self.assertGreater(THEME.section_size, THEME.body_size)

    def test_semantic_colors_are_distinct(self):
        from tools.lecture_infographics.theme import THEME
        self.assertNotEqual(THEME.success, THEME.danger)
        self.assertNotEqual(THEME.accent, THEME.primary)


class ComponentTests(unittest.TestCase):
    def test_fit_text_never_goes_below_minimum(self):
        from tools.lecture_infographics.components import fit_text
        image = Image.new("RGB", (600, 200), "white")
        size = fit_text(ImageDraw.Draw(image), "机器人系统架构", (0, 0, 600, 200), 36, 24)
        self.assertGreaterEqual(size, 24)

    def test_fit_text_enforces_theme_floor_when_caller_passes_lower_minimum(self):
        from tools.lecture_infographics.components import fit_text

        image = Image.new("RGB", (180, 100), "white")
        with self.assertRaises(ValueError):
            fit_text(
                ImageDraw.Draw(image),
                "机器人系统架构",
                (0, 0, 180, 100),
                36,
                18,
            )

    def test_fit_text_raises_when_text_cannot_fit_at_24(self):
        from tools.lecture_infographics.components import fit_text

        image = Image.new("RGB", (80, 40), "white")
        with self.assertRaises(ValueError):
            fit_text(
                ImageDraw.Draw(image),
                "机器人系统架构",
                (0, 0, 80, 40),
                36,
                24,
            )


class RenderTests(unittest.TestCase):
    def test_render_figure_writes_1920_by_1080_png(self):
        figures = load_manifest()
        try:
            from tools.lecture_infographics.render import render_figure
        except ImportError as exc:
            self.fail(f"render module must exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "figure.png"
            render_figure(figures["1-1"], Path(tmp), output)
            self.assertTrue(output.exists())
            with Image.open(output) as image:
                self.assertEqual(image.size, (1920, 1080))
                self.assertEqual(image.format, "PNG")
                self.assertIn(image.mode, {"RGB", "RGBA"})
                self.assertIn("icc_profile", image.info)

    def test_all_manifest_figures_render_without_omitting_text(self):
        from tools.lecture_infographics.render import render_figure

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for key, spec in load_manifest().items():
                with self.subTest(key=key):
                    render_figure(spec, output_dir, output_dir / f"{key}.png")


if __name__ == "__main__":
    unittest.main()
