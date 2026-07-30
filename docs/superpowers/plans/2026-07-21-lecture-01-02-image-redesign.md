# 第一、二讲配图重做实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成并替换第一讲 9 张配图，为第二讲新增 5 张配图，全部采用统一的 1920×1080 教学信息图风格。

**Architecture:** 使用内置图像生成工具制作无文字的半写实机器人教学插图，再用 Pillow 确定性绘制中文标题、卡片、编号、箭头和结论条。所有最终图片从一份结构化清单生成，避免生成模型造成错别字、乱码和流程关系漂移。

**Tech Stack:** 内置 `image_gen`、Python 3、Pillow、macOS PingFang 中文字体、Markdown、Git。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-07-21-lecture-01-02-image-redesign.md`。
- 最终文件必须是 sRGB PNG、1920×1080、严格 16:9。
- 用户提供的三张图片只作为白底、蓝色标题、黄色强调线、圆角信息框和半写实机器人插图的风格参考。
- 第一讲输出 `figure-1-1.png` 至 `figure-1-9.png`；第二讲输出 `figure-2-1.png` 至 `figure-2-5.png`。
- 中文、英文节点名、步骤编号和结论必须来自清单，不由图像模型自由生成。
- 不提交候选图、缓存、临时裁切文件和 `Unconfirmed 915868.crdownload`。
- 在 `feat/part1` 当前分支原位执行；用户已连续要求在该分支修改并提交，不另建 worktree。

---

### Task 1: 建立可验证的配图清单与渲染骨架

**Files:**
- Create: `tools/lecture_infographics/manifest.py`
- Create: `tools/lecture_infographics/render.py`
- Create: `tools/lecture_infographics/test_render.py`

**Interfaces:**
- `manifest.FIGURES: dict[str, FigureSpec]`：14 张图的唯一内容来源。
- `render.render_figure(spec: FigureSpec, asset_dir: Path, out_path: Path) -> None`：渲染一张 1920×1080 PNG。
- `render.render_all(asset_dir: Path, repo_root: Path) -> list[Path]`：渲染 14 张正式图片。

- [ ] **Step 1: 编写失败测试**

测试必须断言：清单恰好包含 14 张图；编号为 1-1…1-9、2-1…2-5；类型只允许 A/B/C；第一讲没有 C 类；B 类步骤编号连续；每张图的标题和结论非空。

Run:

```bash
RUNTIME_PY=/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
$RUNTIME_PY -m unittest tools.lecture_infographics.test_render -v
```

Expected: FAIL，因为 `manifest.py` 和 `render.py` 尚不存在。

- [ ] **Step 2: 实现清单数据结构**

使用 `dataclass(frozen=True)` 定义：

```python
@dataclass(frozen=True)
class FigureSpec:
    key: str
    lecture: int
    number: int
    template: Literal["A", "B", "C"]
    title: str
    subtitle: str
    takeaway: str
    labels: tuple[str, ...]
    steps: tuple[str, ...] = ()
    asset: str | None = None
```

`FIGURES` 的文字逐字复制设计文档第 4、5 节，不自行改写。

- [ ] **Step 3: 实现通用画布与组件**

`render.py` 必须提供：

```python
CANVAS = (1920, 1080)
FONT_PATH = "/System/Library/Fonts/PingFang.ttc"

def new_canvas() -> Image.Image: ...
def draw_header(draw, title: str, subtitle: str) -> None: ...
def draw_takeaway(draw, text: str) -> None: ...
def draw_card(draw, box, title: str, body: str, accent: str) -> None: ...
def draw_arrow(draw, start, end, color: str, width: int = 8) -> None: ...
def draw_step_badge(draw, center, number: int, color: str) -> None: ...
def paste_asset(canvas, asset_path: Path, box) -> None: ...
```

颜色固定为：深蓝 `#123B8F`、亮蓝 `#1769E0`、黄色 `#F4B400`、绿色 `#13A473`、红色 `#E64B5D`、正文 `#172033`、背景 `#F8FAFE`。

- [ ] **Step 4: 运行单元测试**

Expected: 清单测试 PASS；渲染尺寸测试在临时目录生成 1920×1080 RGB/RGBA PNG 并 PASS。

---

### Task 2: 生成统一的机器人教学插图资产

**Files:**
- Create: `tools/lecture_infographics/generated_assets/robot_arm_workcell.png`
- Create: `tools/lecture_infographics/generated_assets/robot_camera_workcell.png`
- Create: `tools/lecture_infographics/generated_assets/mobile_manipulator.png`
- Create: `tools/lecture_infographics/generated_assets/humanoid_robot.png`
- Create: `tools/lecture_infographics/generated_assets/ai_digital_world.png`
- Create: `tools/lecture_infographics/generated_assets/robot_task_scene.png`

**Interfaces:** 每张资产是无文字、纯白背景的教学插图，可直接贴入浅色信息图画布。

- [ ] **Step 1: 生成机械臂工作单元**

Prompt:

```text
Use case: scientific-educational
Asset type: reusable robot illustration for a Chinese robotics textbook infographic
Input image: the user's A-template hand-eye-calibration image is style reference only
Primary request: a clean semi-realistic 3D educational render of a white six-axis collaborative robot arm on a light gray workbench, with a small green cube, compact depth camera, controller box, emergency-stop button, visible cable connection and power unit
Composition: isolated workcell, three-quarter view, all hardware fully visible, generous white margin
Style: white background, polished educational 3D render, subtle soft shadow, blue accents, technically plausible joints and gripper
Constraints: no text, no labels, no logo, no watermark, no decorative UI, no extra robots
```

- [ ] **Step 2: 生成眼在手上的机械臂场景**

生成一台白色六关节机械臂，夹爪附近安装小型相机，观察红色杯子和蓝色托盘；纯白背景；无文字、品牌、水印和额外机械臂。

- [ ] **Step 3: 生成移动操作机器人**

生成轮式移动底盘、白色机械臂和前向深度相机组成的移动操作机器人；干净白色工作室背景；无文字、品牌和水印。

- [ ] **Step 4: 生成人形机器人**

生成友好的全身白色科研人形机器人，正常关节和中立站姿，蓝色点缀；无人物身份、文字、品牌和水印。

- [ ] **Step 5: 生成数字 AI 场景**

生成一台笔记本电脑、抽象文字/图像输入块、模型核心和输出块；不得出现可读文字、品牌和水印。

- [ ] **Step 6: 生成机器人抓取任务场景**

生成一台白色六关节机械臂、红色杯子、托盘和一个可移动障碍块，使左侧抓取、上方抓取和先移开障碍三种候选动作便于后续标注；无箭头、文字、品牌和水印。

- [ ] **Step 7: 检查资产**

使用 `view_image` 逐张确认机器人结构、相机位置、物体数量和白底；出现品牌、文字、额外机械臂或错误肢体时，只针对该问题重生成一次。

---

### Task 3: 渲染第一讲 9 张 A/B 类图片

**Files:**
- Modify: `tools/lecture_infographics/render.py`
- Replace: `docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-1.png` … `figure-1-9.png`

**Interfaces:** `render_lecture_one()` 消费 `FIGURES` 和 Task 2 资产，输出 9 张正式 PNG。

- [ ] **Step 1: 实现 A 类布局**

支持三种构图：左右对比、中心辐射、阶梯/并列卡片。A 类不显示步骤编号，标签不超过设计清单数量。

- [ ] **Step 2: 实现 B 类布局**

支持水平编号流程与环形编号流程；编号从 1 开始连续；闭环必须有绿色回流箭头。

- [ ] **Step 3: 为图 1-1…1-9 建立布局映射**

映射固定为：1-1 左右对比；1-2 中心辐射；1-3 水平闭环；1-4 五卡片；1-5 输入融合输出；1-6 三候选预测；1-7 环形流程；1-8 难度阶梯；1-9 水平闭环。

- [ ] **Step 4: 渲染并检查尺寸**

```bash
$RUNTIME_PY tools/lecture_infographics/render.py --lecture 1
$RUNTIME_PY -m unittest tools.lecture_infographics.test_render -v
```

Expected: 9 张正式文件存在，均为 1920×1080 PNG。

- [ ] **Step 5: 视觉检查**

制作 3×3 联系表并用 `view_image` 检查：标题未裁切、文字无溢出、编号连续、箭头正确、图 1-9 明确写出“二维闭环示意，不是机械臂仿真”。

---

### Task 4: 渲染第二讲 5 张图片并接入正文

**Files:**
- Modify: `tools/lecture_infographics/render.py`
- Create: `docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-1.png` … `figure-2-5.png`
- Modify: `docs/part1-system-basics/02-ros2-architecture.md`

**Interfaces:** `render_lecture_two()` 输出 5 张正式 PNG；Markdown 使用相对路径 `part1_picture/Lecture2_picture/figure-2-N.png`。

- [ ] **Step 1: 实现 C 类架构布局**

图 2-2 使用上层软件/中层接口/下层硬件三层结构，左侧绿色状态上行、右侧蓝色指令下行，右侧接口契约框包含六项准确文字。图 2-4 使用 Node/通信机制/Launch 三分区。

- [ ] **Step 2: 实现第二讲 A/B 布局**

图 2-1 为硬件引线示意；图 2-3 为上下行双向编号流程；图 2-5 为六节点数据流，目标和状态共同指向策略节点。

- [ ] **Step 3: 渲染并检查尺寸**

Expected: 5 张正式文件存在，均为 1920×1080 PNG。

- [ ] **Step 4: 更新第二讲 Markdown**

按设计插入：图 2-1 于 2.2.1；图 2-2 于 2.2 开头；图 2-3 于 2.2.5；图 2-4 于 2.3.7 后；图 2-5 于 2.6.1/2.6.2。每张图后紧跟斜体图注。

- [ ] **Step 5: 视觉检查**

制作第二讲联系表并检查 C 类图在正文宽度下仍可阅读，接口契约和 ROS2 名称无错字。

---

### Task 5: 全量验收并提交

**Files:**
- Verify: 第一、二讲 Markdown、14 张正式 PNG、渲染工具和生成资产。

- [ ] **Step 1: 运行自动验证**

```bash
$RUNTIME_PY -m unittest tools.lecture_infographics.test_render -v
git diff --check
```

验证 14 张图的尺寸、模式、文件名、Markdown 引用和清单覆盖。

- [ ] **Step 2: 全量视觉复核**

用 `view_image` 检查 14 张最终图和两张联系表。发现单张问题时只修改该图，不整体换风格。

- [ ] **Step 3: 检查暂存范围**

明确排除 `.crdownload`、联系表、临时候选图和生成缓存。

- [ ] **Step 4: 创建提交**

```bash
git add docs/part1-system-basics/02-ros2-architecture.md \
  docs/part1-system-basics/part1_picture/Lecture1_picture/figure-1-*.png \
  docs/part1-system-basics/part1_picture/Lecture2_picture/figure-2-*.png \
  tools/lecture_infographics \
  docs/superpowers/plans/2026-07-21-lecture-01-02-image-redesign.md
git commit -m "重做第一二讲教学配图"
```
