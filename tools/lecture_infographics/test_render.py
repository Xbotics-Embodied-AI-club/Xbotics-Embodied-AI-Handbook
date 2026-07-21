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

    def test_font_rejects_sizes_below_theme_floor(self):
        from tools.lecture_infographics.components import font
        from tools.lecture_infographics.theme import THEME

        with self.assertRaises(ValueError):
            font(THEME.min_text_size - 1)


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


class AssetTests(unittest.TestCase):
    def test_required_asset_catalog_is_complete(self):
        from tools.lecture_infographics.assets import ASSETS

        self.assertEqual(
            set(ASSETS),
            {
                "robot_arm",
                "robot_arm_camera",
                "desktop_pick",
                "mobile_manipulator",
                "humanoid_robot",
            },
        )

    def test_project_assets_are_large_clean_rgb_images(self):
        from tools.lecture_infographics.assets import ASSETS

        asset_dir = Path("tools/lecture_infographics/generated_assets")
        for spec in ASSETS.values():
            with Image.open(asset_dir / spec.filename) as image:
                self.assertGreaterEqual(image.width, 1024)
                self.assertGreaterEqual(image.height, 768)
                self.assertIn(image.mode, {"RGB", "RGBA"})


class LayoutTests(unittest.TestCase):
    def test_figure_1_6_uses_vector_checkmark_not_font_glyph(self):
        import inspect

        from tools.lecture_infographics.layouts_a import draw_1_6

        source = inspect.getsource(draw_1_6)
        self.assertNotIn("✓", source)
        self.assertIn("draw_checkmark(", source)

    def test_all_a_figures_have_explicit_layouts(self):
        from tools.lecture_infographics.layouts_a import A_LAYOUTS

        figures = load_manifest()
        expected = {key for key, spec in figures.items() if spec.template == "A"}
        self.assertEqual(set(A_LAYOUTS), expected)

    def test_household_stage_has_dedicated_scene_cues(self):
        from tools.lecture_infographics.layouts_a import HOUSEHOLD_SCENE_CUES

        self.assertEqual(
            set(HOUSEHOLD_SCENE_CUES),
            {"sofa", "floor_lamp", "coffee_table"},
        )

    def test_render_module_has_no_legacy_a_layouts(self):
        from tools.lecture_infographics import render

        legacy_names = {
            "generic_a",
            "visual_1_1",
            "visual_1_2",
            "visual_1_4",
            "visual_1_5",
            "visual_1_6",
            "visual_1_8",
            "visual_2_1",
        }
        self.assertTrue(legacy_names.isdisjoint(vars(render)))

    def test_all_b_figures_have_explicit_layouts(self):
        from tools.lecture_infographics.layouts_b import B_LAYOUTS

        figures = load_manifest()
        expected = {key for key, spec in figures.items() if spec.template == "B"}
        self.assertEqual(set(B_LAYOUTS), expected)

    def test_all_c_figures_have_explicit_layouts(self):
        from tools.lecture_infographics.layouts_c import C_LAYOUTS

        self.assertEqual(set(C_LAYOUTS), {"2-2", "2-4"})

    def test_every_manifest_key_has_one_layout(self):
        from tools.lecture_infographics.layouts_a import A_LAYOUTS
        from tools.lecture_infographics.layouts_b import B_LAYOUTS
        from tools.lecture_infographics.layouts_c import C_LAYOUTS

        a_keys, b_keys, c_keys = set(A_LAYOUTS), set(B_LAYOUTS), set(C_LAYOUTS)
        self.assertTrue(a_keys.isdisjoint(b_keys))
        self.assertTrue(a_keys.isdisjoint(c_keys))
        self.assertTrue(b_keys.isdisjoint(c_keys))
        assigned = a_keys | b_keys | c_keys
        self.assertEqual(assigned, set(load_manifest()))
        self.assertEqual(len(assigned), 14)

    def test_c_lane_text_and_arrow_columns_are_separated(self):
        from tools.lecture_infographics.layouts_c import LANE_LAYOUTS

        self.assertEqual(set(LANE_LAYOUTS), {"state_up", "command_down"})
        for name, lane in LANE_LAYOUTS.items():
            with self.subTest(name=name):
                x1, _, x2, _ = lane.box
                self.assertGreater(lane.text_x, x1)
                self.assertLess(lane.text_x, x2)
                self.assertGreater(lane.arrow_x, x1)
                self.assertLess(lane.arrow_x, x2)
                self.assertGreaterEqual(abs(lane.text_x - lane.arrow_x), 40)

    def test_all_b_figures_declare_rendered_loop_routes(self):
        from tools.lecture_infographics.layouts_b import LOOP_ROUTES

        figures = load_manifest()
        expected = {key for key, spec in figures.items() if spec.template == "B"}
        self.assertEqual(set(LOOP_ROUTES), expected)
        for key, routes in LOOP_ROUTES.items():
            self.assertGreaterEqual(len(routes), 1, key)
            for route in routes:
                self.assertIn(route.semantic, {"success", "danger"}, (key, route.name))

    def test_failure_samples_return_to_filtering_and_training(self):
        from tools.lecture_infographics.layouts_b import LOOP_ROUTES

        failure_routes = [
            route
            for route in LOOP_ROUTES["1-7"]
            if route.semantic == "danger"
        ]
        self.assertEqual(len(failure_routes), 1)
        self.assertEqual(failure_routes[0].target, "filter_training")

    def test_joint_states_returns_to_policy_and_status(self):
        from tools.lecture_infographics.layouts_b import LOOP_ROUTES

        joint_state_routes = [
            route
            for route in LOOP_ROUTES["2-5"]
            if route.semantic == "success" and route.topic == "/joint_states"
        ]
        self.assertEqual(
            {route.target for route in joint_state_routes},
            {"policy_node", "task_status_node"},
        )

    def test_b_layouts_draw_every_declared_loop_route(self):
        from unittest.mock import patch

        from tools.lecture_infographics.layouts_b import B_LAYOUTS, LOOP_ROUTES

        image = Image.new("RGB", (1920, 1080), "white")
        draw = ImageDraw.Draw(image)
        for key, routes in LOOP_ROUTES.items():
            with self.subTest(key=key):
                with patch(
                    "tools.lecture_infographics.layouts_b._draw_loop_route"
                ) as draw_loop_route:
                    B_LAYOUTS[key](image, draw, load_manifest()[key], Path("."))
                rendered_names = {
                    call.args[2] for call in draw_loop_route.call_args_list
                }
                self.assertEqual(
                    rendered_names,
                    {route.name for route in routes},
                )

    def test_loop_route_endpoints_resolve_to_node_anchors(self):
        from tools.lecture_infographics.layouts_b import LOOP_ROUTES, NODE_ANCHORS

        self.assertEqual(set(NODE_ANCHORS), set(LOOP_ROUTES))
        for key, routes in LOOP_ROUTES.items():
            for route in routes:
                self.assertIn(route.source, NODE_ANCHORS[key], (key, route.name))
                self.assertIn(route.target, NODE_ANCHORS[key], (key, route.name))
                self.assertNotIn(NODE_ANCHORS[key][route.source], route.points)
                self.assertNotIn(NODE_ANCHORS[key][route.target], route.points)

    def test_draw_loop_route_uses_resolved_source_and_target_anchors(self):
        from unittest.mock import patch

        from tools.lecture_infographics.layouts_b import (
            LOOP_ROUTES,
            NODE_ANCHORS,
            _draw_loop_route,
        )

        image = Image.new("RGB", (1920, 1080), "white")
        draw = ImageDraw.Draw(image)
        for key, routes in LOOP_ROUTES.items():
            for route in routes:
                with self.subTest(key=key, route=route.name):
                    with patch(
                        "tools.lecture_infographics.layouts_b._path_arrow"
                    ) as path_arrow:
                        _draw_loop_route(draw, key, route.name)
                    path_arrow.assert_called_once_with(
                        draw,
                        (
                            NODE_ANCHORS[key][route.source],
                            *route.points,
                            NODE_ANCHORS[key][route.target],
                        ),
                        semantic=route.semantic,
                        width=7,
                    )

    def test_critical_loop_route_metadata_is_exact(self):
        from tools.lecture_infographics.layouts_b import LOOP_ROUTES

        routes_1_7 = {route.name: route for route in LOOP_ROUTES["1-7"]}
        failure = routes_1_7["failure_to_filter_training"]
        self.assertEqual(
            (failure.source, failure.target, failure.semantic, failure.topic),
            ("evaluation", "filter_training", "danger", "失败样本"),
        )

        routes_2_5 = {route.name: route for route in LOOP_ROUTES["2-5"]}
        policy = routes_2_5["joint_states_to_policy"]
        status = routes_2_5["joint_states_to_status"]
        self.assertEqual(
            (policy.source, policy.target, policy.semantic, policy.topic),
            ("robot_state_node", "policy_node", "success", "/joint_states"),
        )
        self.assertEqual(
            (status.source, status.target, status.semantic, status.topic),
            ("robot_state_node", "task_status_node", "success", "/joint_states"),
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
