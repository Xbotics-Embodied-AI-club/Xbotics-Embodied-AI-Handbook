from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image


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


if __name__ == "__main__":
    unittest.main()
