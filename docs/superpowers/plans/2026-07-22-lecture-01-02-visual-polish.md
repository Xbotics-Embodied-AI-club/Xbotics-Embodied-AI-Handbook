# 第一、二讲配图视觉美化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 14 张图知识内容、图号和正文引用的前提下，统一字体、配色、组件与机器人素材，输出更精致且可重复生成的 16:9 教学配图。

**Architecture:** 将当前单文件渲染器拆为视觉主题、通用组件、素材加载和 A/B/C 布局四类职责；内容继续由 `manifest.py` 提供，`render.py` 只负责调度。AI 生成无文字 3D 主体素材，Pillow 负责全部文字、图标、流程、图层和最终 sRGB 输出。

**Tech Stack:** Python 3.12、Pillow、内置 `image_gen`、`unittest`、Git。

## Global Constraints

- 全部成图必须为 1920×1080、16:9 横版、RGB/RGBA、嵌入 sRGB 配置的 PNG。
- 保持 `manifest.py` 中的知识内容、图号、标题和核心结论不变。
- 一级标题 64–68 px；副标题 28–30 px；模块标题与关键结论 30–36 px；正文 26–30 px；必要文字不得低于 24 px。
- 绿色只表示正确、反馈、通过或安全路径；红色只表示风险、错误或不通过；黄色只用于重点、序号或注意提示。
- AI 只生成无文字、无品牌、无水印主体素材，全部文字和接口名由 Pillow 绘制。
- 不引入外部在线字体或运行时网络依赖。
- 不改动用户文件 `docs/part1-system-basics/part1_picture/Lecture1_picture/Unconfirmed 915868.crdownload`。

---

## File Map

- Create `tools/lecture_infographics/theme.py`: 颜色、字号、线宽、圆角、间距和阴影常量。
- Create `tools/lecture_infographics/components.py`: 标题、结论条、卡片、徽标、箭头、文本适配和图标组件。
- Create `tools/lecture_infographics/assets.py`: 机器人素材目录、校验与白底素材裁切加载。
- Create `tools/lecture_infographics/layouts_a.py`: 图 1-1、1-2、1-4、1-5、1-6、1-8、2-1。
- Create `tools/lecture_infographics/layouts_b.py`: 图 1-3、1-7、1-9、2-3、2-5。
- Create `tools/lecture_infographics/layouts_c.py`: 图 2-2、2-4。
- Modify `tools/lecture_infographics/render.py`: 只保留画布创建、模板调度和保存。
- Modify `tools/lecture_infographics/test_render.py`: 主题、素材、布局、输出和链接测试。
- Create `tools/lecture_infographics/generated_assets/robot-arm-camera.png`.
- Create `tools/lecture_infographics/generated_assets/desktop-pick-scene.png`.
- Create `tools/lecture_infographics/generated_assets/mobile-manipulator.png`.
- Create `tools/lecture_infographics/generated_assets/humanoid-robot.png`.
- Modify `tools/lecture_infographics/generated_assets/robot-arm.png` only if新版本在统一性和边缘质量上明显优于当前素材。
- Modify `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-1.png` through `figure-1-9.png`.
- Modify `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-1.png` through `figure-2-5.png`.

---

### Task 1: 建立统一视觉主题与组件

**Files:**
- Create: `tools/lecture_infographics/theme.py`
- Create: `tools/lecture_infographics/components.py`
- Modify: `tools/lecture_infographics/test_render.py`
- Modify: `tools/lecture_infographics/render.py`

**Interfaces:**
- Produces: `THEME: Theme`; `font(size: int, weight: str = "regular")`; `fit_text(draw, text, box, max_size, min_size=24, weight="regular", fill=None)`; `draw_header(draw, spec)`; `draw_takeaway(draw, text)`; `draw_card(draw, box, style="default")`; `draw_arrow(draw, start, end, semantic="primary", width=None)`.
- Consumes: `FigureSpec` from `manifest.py` and Pillow drawing primitives.

- [ ] **Step 1: Write failing theme and component tests**

Add these tests:

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render.ThemeTests tools.lecture_infographics.test_render.ComponentTests -v
```

Expected: FAIL because `theme.py` and `components.py` do not exist.

- [ ] **Step 3: Implement theme constants**

Create `theme.py` with this public shape:

```python
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
```

- [ ] **Step 4: Move shared drawing functions into components**

Move and refine the existing font, rounded card, fitted text, header, takeaway and arrow helpers. `fit_text` must return the chosen font size and raise `ValueError` if text cannot fit at `min_size`; it must never silently omit text. Use a soft one-layer shadow offset `(0, 6)` with low-opacity blue-gray for major cards only.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all listed tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/lecture_infographics/theme.py tools/lecture_infographics/components.py tools/lecture_infographics/render.py tools/lecture_infographics/test_render.py
git commit -m "统一教学配图视觉主题"
```

---

### Task 2: 生成并接入统一机器人素材组

**Files:**
- Create: `tools/lecture_infographics/assets.py`
- Create: `tools/lecture_infographics/generated_assets/robot-arm-camera.png`
- Create: `tools/lecture_infographics/generated_assets/desktop-pick-scene.png`
- Create: `tools/lecture_infographics/generated_assets/mobile-manipulator.png`
- Create: `tools/lecture_infographics/generated_assets/humanoid-robot.png`
- Modify: `tools/lecture_infographics/test_render.py`

**Interfaces:**
- Produces: `ASSETS: dict[str, AssetSpec]`; `load_asset(asset_dir: Path, name: str, max_size: tuple[int, int]) -> Image.Image`.
- Consumes: project-local PNG files with white background and no embedded text.

- [ ] **Step 1: Write failing asset catalog tests**

```python
class AssetTests(unittest.TestCase):
    def test_required_asset_catalog_is_complete(self):
        from tools.lecture_infographics.assets import ASSETS
        self.assertEqual(set(ASSETS), {
            "robot_arm", "robot_arm_camera", "desktop_pick",
            "mobile_manipulator", "humanoid_robot",
        })

    def test_project_assets_are_large_clean_rgb_images(self):
        from tools.lecture_infographics.assets import ASSETS
        asset_dir = Path("tools/lecture_infographics/generated_assets")
        for spec in ASSETS.values():
            with Image.open(asset_dir / spec.filename) as image:
                self.assertGreaterEqual(image.width, 1024)
                self.assertGreaterEqual(image.height, 768)
                self.assertIn(image.mode, {"RGB", "RGBA"})
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render.AssetTests -v
```

Expected: FAIL because the catalog and four new assets are absent.

- [ ] **Step 3: Generate four assets with built-in image generation**

Issue one built-in `image_gen` call per asset. Use this shared style in every prompt:

```text
Use case: scientific-educational
Asset type: Chinese robotics textbook infographic subject
Style/medium: polished semi-realistic 3D educational illustration
Lighting/mood: soft studio lighting, subtle contact shadow, calm and precise
Color palette: white and light-gray robot shells, deep navy joints, restrained blue details
Composition: wide 16:9, centered subject, generous pure-white negative space
Constraints: no words, no letters, no numbers, no logo, no watermark, no UI, no decorative background
```

Append one exact subject per call:

1. `A six-axis industrial robot arm with a compact RGB-D camera mounted near the wrist and a two-finger gripper.`
2. `A tabletop pick-and-place scene with a six-axis robot arm, one colored cube, a tray, and a small RGB-D camera.`
3. `A compact mobile manipulator with a wheeled base, one articulated arm, a camera mast, and a small gripper.`
4. `A friendly research humanoid robot standing neutrally, full body visible, proportioned for a university laboratory.`

Copy selected outputs into the four exact project paths listed above. Keep generated images text-free.

- [ ] **Step 4: Implement asset catalog and loader**

Create this interface:

```python
@dataclass(frozen=True)
class AssetSpec:
    filename: str
    crop: tuple[int, int, int, int] | None = None

ASSETS = {
    "robot_arm": AssetSpec("robot-arm.png", (280, 10, 1450, 930)),
    "robot_arm_camera": AssetSpec("robot-arm-camera.png"),
    "desktop_pick": AssetSpec("desktop-pick-scene.png"),
    "mobile_manipulator": AssetSpec("mobile-manipulator.png"),
    "humanoid_robot": AssetSpec("humanoid-robot.png"),
}
```

`load_asset` must validate the name, convert to RGB, apply the configured crop, preserve aspect ratio and return a thumbnail no larger than `max_size`.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: both asset tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/lecture_infographics/assets.py tools/lecture_infographics/generated_assets tools/lecture_infographics/test_render.py
git commit -m "统一机器人教学插画素材"
```

---

### Task 3: 美化 A 类概念示意图

**Files:**
- Create: `tools/lecture_infographics/layouts_a.py`
- Modify: `tools/lecture_infographics/render.py`
- Modify: `tools/lecture_infographics/test_render.py`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-1.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-2.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-4.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-5.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-6.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-8.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-1.png`

**Interfaces:**
- Produces: `render_a(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None`.
- Consumes: `THEME`, components helpers, `load_asset`, and A-class figure specs.

- [ ] **Step 1: Write failing A-layout dispatch tests**

```python
class LayoutTests(unittest.TestCase):
    def test_all_a_figures_have_explicit_layouts(self):
        from tools.lecture_infographics.layouts_a import A_LAYOUTS
        figures = load_manifest()
        expected = {key for key, spec in figures.items() if spec.template == "A"}
        self.assertEqual(set(A_LAYOUTS), expected)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render.LayoutTests.test_all_a_figures_have_explicit_layouts -v
```

Expected: FAIL because `layouts_a.py` does not exist.

- [ ] **Step 3: Implement seven explicit A layouts**

Create `A_LAYOUTS` with exact keys `1-1`, `1-2`, `1-4`, `1-5`, `1-6`, `1-8`, `2-1`. Each function must use the shared header and takeaway, keep body text at 26 px or larger, and follow the per-figure requirements in the design spec. Use the generated `desktop_pick`, `mobile_manipulator`, `humanoid_robot`, `robot_arm` and `robot_arm_camera` assets where the subject benefits from a realistic robot.

The central dispatcher must be:

```python
A_LAYOUTS = {
    "1-1": draw_1_1,
    "1-2": draw_1_2,
    "1-4": draw_1_4,
    "1-5": draw_1_5,
    "1-6": draw_1_6,
    "1-8": draw_1_8,
    "2-1": draw_2_1,
}

def render_a(image, draw, spec, asset_dir):
    A_LAYOUTS[spec.key](image, draw, spec, asset_dir)
```

- [ ] **Step 4: Run layout and render tests**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render.LayoutTests tools.lecture_infographics.test_render.RenderTests -v
```

Expected: PASS.

- [ ] **Step 5: Render and inspect the seven A images**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m tools.lecture_infographics.generate
```

Open a contact sheet plus full-size `figure-1-2.png`, `figure-1-6.png` and `figure-2-1.png`. Reject any output with tiny text, inconsistent robot materials, empty center, overlapping leaders or weak focal hierarchy.

- [ ] **Step 6: Commit**

```bash
git add tools/lecture_infographics/layouts_a.py tools/lecture_infographics/render.py tools/lecture_infographics/test_render.py docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-1.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-2.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-4.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-5.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-6.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-8.png docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-1.png
git commit -m "美化第一二讲概念示意图"
```

---

### Task 4: 美化 B 类流程与闭环图

**Files:**
- Create: `tools/lecture_infographics/layouts_b.py`
- Modify: `tools/lecture_infographics/render.py`
- Modify: `tools/lecture_infographics/test_render.py`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-3.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-7.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-9.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-3.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-5.png`

**Interfaces:**
- Produces: `render_b(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None` and `B_LAYOUTS` containing all B keys.
- Consumes: theme and component helpers; no generated text inside raster assets.

- [ ] **Step 1: Write failing B-layout and feedback-loop tests**

```python
def test_all_b_figures_have_explicit_layouts(self):
    from tools.lecture_infographics.layouts_b import B_LAYOUTS
    figures = load_manifest()
    expected = {key for key, spec in figures.items() if spec.template == "B"}
    self.assertEqual(set(B_LAYOUTS), expected)

def test_closed_loop_figures_are_marked_as_loops(self):
    from tools.lecture_infographics.layouts_b import LOOP_FIGURES
    self.assertEqual(LOOP_FIGURES, {"1-3", "1-7", "1-9", "2-3", "2-5"})
```

- [ ] **Step 2: Run tests and verify RED**

Run the two new tests. Expected: FAIL because `layouts_b.py` does not exist.

- [ ] **Step 3: Implement five B layouts**

Use exact dispatch keys `1-3`, `1-7`, `1-9`, `2-3`, `2-5`. Give every step a distinct semantic icon rather than a number-only card. Draw a visible return arrow on all five figures; in `1-7` route failure samples back to filtering/training, and in `2-5` route `/joint_states` from robot state back into policy and status nodes. Keep interface names at 24 px or larger.

- [ ] **Step 4: Run B tests and complete render suite**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render -v
```

Expected: all tests PASS.

- [ ] **Step 5: Render and inspect B images**

Regenerate all images. Inspect full-size figures 1-3, 1-7, 2-3 and 2-5. Verify the reading order is unambiguous, return arrows do not cross text, semantic colors are correct and the smallest English interface text remains readable.

- [ ] **Step 6: Commit**

```bash
git add tools/lecture_infographics/layouts_b.py tools/lecture_infographics/render.py tools/lecture_infographics/test_render.py docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-3.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-7.png docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-9.png docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-3.png docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-5.png
git commit -m "美化机器人流程与闭环配图"
```

---

### Task 5: 美化 C 类原理图并完成全套验收

**Files:**
- Create: `tools/lecture_infographics/layouts_c.py`
- Modify: `tools/lecture_infographics/render.py`
- Modify: `tools/lecture_infographics/test_render.py`
- Modify: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-2.png`
- Modify: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-4.png`

**Interfaces:**
- Produces: `render_c(image: Image.Image, draw: ImageDraw.ImageDraw, spec: FigureSpec, asset_dir: Path) -> None`; final `render_figure` dispatch covering all 14 specs.
- Consumes: theme, components, A/B renderers and manifest.

- [ ] **Step 1: Write failing C-layout and full-dispatch tests**

```python
def test_all_c_figures_have_explicit_layouts(self):
    from tools.lecture_infographics.layouts_c import C_LAYOUTS
    self.assertEqual(set(C_LAYOUTS), {"2-2", "2-4"})

def test_every_manifest_key_has_one_layout(self):
    from tools.lecture_infographics.layouts_a import A_LAYOUTS
    from tools.lecture_infographics.layouts_b import B_LAYOUTS
    from tools.lecture_infographics.layouts_c import C_LAYOUTS
    assigned = set(A_LAYOUTS) | set(B_LAYOUTS) | set(C_LAYOUTS)
    self.assertEqual(assigned, set(load_manifest()))
    self.assertEqual(len(assigned), 14)
```

- [ ] **Step 2: Run tests and verify RED**

Run both new tests. Expected: FAIL because `layouts_c.py` is absent.

- [ ] **Step 3: Implement figure 2-2 architecture layout**

Use three full-width layer bands for application/intelligence, system/control and hardware/execution. Place green state-up and amber command-down lanes in dedicated gutters so they never overlap module labels. Keep the interface-contract sidebar with six numbered checks. Use 28–30 px for layer names, 24–28 px for module names and 25–28 px for contract items.

- [ ] **Step 4: Implement figure 2-4 ROS2 semantics layout**

Give Topic, Service, Action and Launch distinct icons and equal-width cards. Use the lower panel to show `消息类型 → 单位 → 频率 → 时间` as a validation chain, preceded by “通信连通 ≠ 系统正确”. Do not use number-only circles as the primary visual.

- [ ] **Step 5: Run full automated verification**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m tools.lecture_infographics.generate
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tools.lecture_infographics.test_render -v
```

Then run a Python validation that asserts exactly 14 output files, each with size `(1920, 1080)`, mode `RGB` or `RGBA`, and an `icc_profile` entry. Validate that second-lecture Markdown contains exactly five existing `figure-2-[1-5].png` links.

- [ ] **Step 6: Perform visual QA**

Create a 4-column contact sheet of all 14 images. Inspect it at overview scale for consistency, then inspect figures 1-3, 1-7, 2-1, 2-2, 2-3 and 2-5 at original resolution. Verify typography hierarchy, text floor, asset consistency, alignment, whitespace, flow direction and semantic colors against the design spec.

- [ ] **Step 7: Run Git checks and commit**

Run:

```bash
git diff --check
git status --short
```

Stage only the two C images, renderer/layout/test files and any A/B images changed during final visual corrections. Exclude the `.crdownload` file. Commit:

```bash
git commit -m "完成第一二讲配图视觉美化"
```
