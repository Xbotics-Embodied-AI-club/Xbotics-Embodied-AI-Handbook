# 第一、二讲审校问题修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复当前第一、二讲中经审核意见验证仍存在的七项问题，并用自动检查锁定图号、关键措辞、资源信息和 Episode 元数据。

**Architecture:** 采用定点正文修订，不重写章节结构。一个独立 `unittest` 文件读取两篇 Markdown，负责审校问题的内容回归和本地图片引用检查；第一讲与第二讲分别完成红—绿测试循环并独立提交。

**Tech Stack:** Markdown、Python 3 标准库 `unittest`、正则表达式、Git。

## Global Constraints

- 只修改第一讲、第二讲、审校回归测试和本计划文件。
- 不修改审核 DOCX、现有 PNG 内容、其他章节或未跟踪文件。
- `figure-1-7.png` 移至 1.3.7，不重命名图片。
- 推荐资源信息核验日期固定为 2026-07-31。
- 当前关节空间 mock 不虚构相机标定；有空间位姿或相机时才要求 frame 与标定引用。
- 每次提交只暂存计划明确列出的文件。

---

## File Map

- Create `tools/lecture_infographics/test_chapter_review_fixes.py`: 对第一、二讲进行内容和图片引用回归检查。
- Modify `docs/part1-system-basics/01-introduction.md`: 修复定义、数据来源、控制直觉、人形表述、图号顺序和资源表。
- Modify `docs/part1-system-basics/02-ros2-architecture.md`: 增加 Episode 级元数据并重组教学 JSON。

### Task 1: 第一讲审校问题回归与正文修复

**Files:**
- Create: `tools/lecture_infographics/test_chapter_review_fixes.py`
- Modify: `docs/part1-system-basics/01-introduction.md:98-146,218-226,240-249,323-370,484-504`

**Interfaces:**
- Consumes: 第一讲 Markdown 中的标题、图片引用、资源表和正文措辞。
- Produces: `ChapterReviewFixTests`，后续任务在同一测试类中追加第二讲断言。

- [ ] **Step 1: 写入第一讲失败测试**

创建以下测试文件：

```python
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LECTURE_1 = ROOT / "docs/part1-system-basics/01-introduction.md"
LECTURE_2 = ROOT / "docs/part1-system-basics/02-ros2-architecture.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ChapterReviewFixTests(unittest.TestCase):
    def test_lecture_1_reviewed_wording_is_removed(self):
        text = read(LECTURE_1)
        rejected = (
            "完成开放世界任务的能力",
            "数据来自真实机器人交互和仿真",
            "这就是很多基础控制器的核心思想",
            "理论上最适合进入人类环境",
        )
        for phrase in rejected:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, text)

    def test_lecture_1_figures_first_appear_in_numeric_order(self):
        text = read(LECTURE_1)
        figure_numbers = [
            int(value)
            for value in re.findall(r"^!\[图 1-(\d+)", text, flags=re.MULTILINE)
        ]
        self.assertEqual(figure_numbers, list(range(1, 10)))

    def test_lecture_1_resources_are_linked_and_attributed(self):
        text = read(LECTURE_1)
        self.assertIn("维护方", text)
        self.assertIn("版本/分支说明", text)
        self.assertIn("核验日期：2026-07-31", text)
        urls = (
            "https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook",
            "https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent",
            "https://github.com/GT-RIPL/Awesome-LLM-Robotics",
            "https://github.com/tsinghua-fib-lab/World-Model",
            "https://github.com/huggingface/lerobot",
            "https://github.com/openvla/openvla",
            "https://developer.nvidia.com/isaac/lab",
            "https://maniskill.readthedocs.io/en/latest/",
            "https://github.com/YanjieZe/awesome-humanoid-robot-learning",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertIn(url, text)

    def test_lecture_1_local_images_exist(self):
        text = read(LECTURE_1)
        targets = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
        local_targets = [target for target in targets if "://" not in target]
        self.assertGreater(len(local_targets), 0)
        for target in local_targets:
            with self.subTest(target=target):
                self.assertTrue((LECTURE_1.parent / target).is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认针对旧正文失败**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tools.lecture_infographics.test_chapter_review_fixes -v
```

Expected: `test_lecture_1_reviewed_wording_is_removed`、`test_lecture_1_figures_first_appear_in_numeric_order` 和 `test_lecture_1_resources_are_linked_and_attributed` 失败；本地图片检查通过。

- [ ] **Step 3: 修复定义与数据来源**

把定义改为：

```markdown
> 具身智能是指智能体依托具体身体，在真实或仿真的环境中，通过感知—行动闭环与环境交互，并在物理约束下完成任务和适应变化的能力。

开放世界泛化是具身智能的重要研究目标，但不是具身智能成立的定义前提。工业装配、固定工位抓取等结构化任务，同样体现身体、行动、反馈和环境交互。
```

把数据来源对比行改为：

```markdown
| 常大量使用互联网文本、图像和视频 | 除互联网视觉语言数据外，还依赖示教、机器人交互、部署日志、仿真及合成数据 |
```

- [ ] **Step 4: 修复控制直觉与人形表述**

用以下两段替换原控制概括：

```markdown
> 对最简单的比例反馈控制，可以让控制量随位置误差增大，并在接近目标时逐渐减小。

这只是比例反馈的入门直觉。工程实现还必须设置增益、限幅、采样周期和停止条件，并由底层伺服与安全约束保证动作可执行；更复杂的控制器还会考虑速度、加速度、阻尼、动力学和时延。
```

把人形比较改为：

```markdown
机械臂擅长桌面抓取和固定工位操作，但很难自己移动到另一个房间。移动底盘可以导航，但没有操作机构就无法抓取物体。人形结构可能兼容部分按人体尺度设计的设施，减少某些场景的环境改造成本；是否适用仍取决于尺寸、负载、机动性、稳定性、安全和具体任务需求。
```

- [ ] **Step 5: 移动图 1-7**

把以下图片和图注从 1.8 节删除：

```markdown
![图 1-7 数据飞轮：机器人怎样越用越好](part1_picture/Lecture1_picture/figure-1-7.png)

*图 1-7 数据飞轮：机器人怎样越用越好*
```

将其插入 1.3.7“数据记录与模型迭代”的开头说明之后、数据飞轮流程展开之前。不要移动或重命名 `figure-1-8.png` 和 `figure-1-9.png`。

- [ ] **Step 6: 重建推荐资源表**

表前增加：

```markdown
以下链接与维护信息核验于 2026-07-31。持续更新的资料列表按 `main` 分支跟踪；软件框架的具体实验版本应以对应课程环境或 lockfile 为准。
```

使用以下表格：

```markdown
| 类型 | 资源及官方链接 | 维护方 | 版本/分支说明 | 用途 |
| --- | --- | --- | --- | --- |
| 具身智能总览 | [Xbotics-Embodied-AI-Handbook](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook) | Xbotics Embodied AI Club | 随书稿版本同步 | 建立中文学习路线和知识地图 |
| 前沿论文与项目 | [Awesome-Embodied-Robotics-and-Agent](https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent) | zchoi 社区维护 | `main` | 跟踪 VLM、LLM、VLA 和具身 Agent |
| LLM/VLM + Robotics | [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) | GT-RIPL | `main` | 理解大模型在机器人规划与交互中的应用 |
| 世界模型 | [Awesome-World-Model](https://github.com/tsinghua-fib-lab/World-Model) | Tsinghua FIB Lab | `main` | 学习世界模型综述、论文和项目资源 |
| 真实机器人学习 | [LeRobot](https://github.com/huggingface/lerobot) | Hugging Face LeRobot Team | 官方 releases；实验需锁定版本 | 学习数据采集、数据格式和模仿学习训练 |
| VLA 模型 | [OpenVLA](https://github.com/openvla/openvla) | OpenVLA Team | `main`；依赖按仓库说明锁定 | 理解 VLA 结构、推理与微调 |
| 仿真训练 | [Isaac Lab](https://developer.nvidia.com/isaac/lab) | NVIDIA | 官方文档与 releases | 学习强化学习、人形机器人和 Sim2Real 实验 |
| 操作 Benchmark | [ManiSkill](https://maniskill.readthedocs.io/en/latest/) | ManiSkill Team / Hao Su Lab | 官方稳定文档 | 学习桌面操作、移动操作和机器人学习评测 |
| 人形机器人 | [Awesome-Humanoid-Robot-Learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) | Yanjie Ze 社区维护 | `main` | 跟踪人形运动控制、操作与 Sim2Real |
```

- [ ] **Step 7: 运行第一讲测试并确认通过**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tools.lecture_infographics.test_chapter_review_fixes -v
```

Expected: 4 tests pass.

- [ ] **Step 8: 检查第一讲差异并提交**

Run:

```bash
git diff --check -- docs/part1-system-basics/01-introduction.md tools/lecture_infographics/test_chapter_review_fixes.py
git diff -- docs/part1-system-basics/01-introduction.md tools/lecture_infographics/test_chapter_review_fixes.py
git add docs/part1-system-basics/01-introduction.md tools/lecture_infographics/test_chapter_review_fixes.py
git commit -m "修复第一讲审校问题"
```

Expected: 只提交第一讲正文与新测试文件。

### Task 2: 第二讲 Episode 元数据回归与正文修复

**Files:**
- Modify: `tools/lecture_infographics/test_chapter_review_fixes.py`
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:449-480`

**Interfaces:**
- Consumes: Task 1 创建的 `ChapterReviewFixTests`、`read()`、`LECTURE_2`。
- Produces: Episode 级 metadata 和 step 级时序记录的明确契约。

- [ ] **Step 1: 增加第二讲失败测试**

在 `ChapterReviewFixTests` 中增加：

```python
    def test_lecture_2_episode_metadata_is_reproducible(self):
        text = read(LECTURE_2)
        required = (
            "schema_version",
            "robot_model",
            "firmware_version",
            "software_revision",
            "joint_names",
            "joint_unit",
            "target_mode",
            "action_mode",
            "nominal_control_hz",
            "task_id",
            "reset_condition",
            "calibration_refs",
        )
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, text)

    def test_lecture_2_local_images_exist(self):
        text = read(LECTURE_2)
        targets = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
        local_targets = [target for target in targets if "://" not in target]
        self.assertGreater(len(local_targets), 0)
        for target in local_targets:
            with self.subTest(target=target):
                self.assertTrue((LECTURE_2.parent / target).is_file())
```

- [ ] **Step 2: 运行测试并确认 metadata 测试失败**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tools.lecture_infographics.test_chapter_review_fixes -v
```

Expected: `test_lecture_2_episode_metadata_is_reproducible` 失败；第二讲本地图片检查通过。

- [ ] **Step 3: 把 Episode 记录拆成 metadata 与 steps**

在“一个最小记录应包含”之前改为：

```markdown
要让记录可复现，Episode 应分成两个层级：Episode 级元数据说明整段数据采用什么机器人、软件和接口语义；Step 级数据记录每个时刻发生了什么。只保存数值而不保存口径，后续回放和训练仍可能读错数据。

Episode 级元数据至少包括：

| 字段 | 含义 | 为什么需要 |
| --- | --- | --- |
| `schema_version` | 记录格式版本 | 让读取程序知道字段结构与兼容规则 |
| `robot_model` | 机器人或 mock 模型 | 区分不同本体和自由度 |
| `firmware_version`、`software_revision` | 固件和控制软件版本 | 追溯接口或行为变化 |
| `joint_names`、`joint_unit` | 关节顺序和单位 | 防止数组错位或角度/弧度混用 |
| `target_mode`、`action_mode` | 目标和动作是绝对量、增量、速度还是其他形式 | 保证动作语义一致 |
| `nominal_control_hz` | 名义控制频率 | 解释每步动作对应的时间尺度 |
| `task_id`、`reset_condition` | 任务定义和重置条件 | 区分任务边界并复现实验起点 |
| `frame_id`、`calibration_refs` | 空间坐标系和传感器标定引用 | 接入末端位姿、相机或外部传感器时保持几何语义一致 |

当前 Demo 是不带相机的关节空间 mock，因此 `frame_id` 可以为空，`calibration_refs` 可以是空列表；但关节名称、顺序、单位、动作语义和软件版本不能省略。接入末端位姿、相机图像或真实硬件后，必须填写对应 frame、标定版本和硬件版本。

每个 Step 至少包含：
```

保留原有 Step 字段表。

- [ ] **Step 4: 用两层 JSON 替换原教学记录**

使用以下示例：

```json
{
  "metadata": {
    "schema_version": "robot_demo_episode/v1",
    "robot_model": "teaching_mock_6dof",
    "firmware_version": "mock",
    "software_revision": "robot_demo@1.0.0",
    "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "joint_unit": "rad",
    "target_mode": "absolute_joint_position",
    "action_mode": "delta_joint_position",
    "nominal_control_hz": 20,
    "task_id": "reach_joint_target",
    "reset_condition": "all_joint_positions_zero",
    "frame_id": null,
    "calibration_refs": []
  },
  "steps": [
    {
      "timestamp": 1721102401.42,
      "joint_state": [0.00, -0.31, 0.62, 0.00, 0.28, 0.00],
      "target_joint": [0.00, -0.40, 0.80, 0.00, 0.40, 0.00],
      "action_command": [0.00, -0.045, 0.05, 0.00, 0.05, 0.00],
      "task_status": "running",
      "success": false,
      "error": null
    }
  ]
}
```

- [ ] **Step 5: 运行第二讲测试并确认通过**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tools.lecture_infographics.test_chapter_review_fixes -v
```

Expected: 6 tests pass.

- [ ] **Step 6: 检查第二讲差异并提交**

Run:

```bash
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md tools/lecture_infographics/test_chapter_review_fixes.py
git diff -- docs/part1-system-basics/02-ros2-architecture.md tools/lecture_infographics/test_chapter_review_fixes.py
git add docs/part1-system-basics/02-ros2-architecture.md tools/lecture_infographics/test_chapter_review_fixes.py
git commit -m "完善第二讲 Episode 元数据"
```

Expected: 只提交第二讲正文和测试增量。

### Task 3: 全量验收

**Files:**
- Verify: `docs/part1-system-basics/01-introduction.md`
- Verify: `docs/part1-system-basics/02-ros2-architecture.md`
- Verify: `tools/lecture_infographics/test_chapter_review_fixes.py`

**Interfaces:**
- Consumes: Task 1 和 Task 2 的正文与测试提交。
- Produces: 可交付的验证证据和干净的提交边界。

- [ ] **Step 1: 运行审校回归测试与现有配图测试**

Run:

```bash
/Users/shadow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest \
  tools.lecture_infographics.test_chapter_review_fixes \
  tools.lecture_infographics.test_render -v
```

Expected: 新增 6 项审校测试和现有 28 项配图测试全部通过。

- [ ] **Step 2: 检查 Markdown 与 Git 边界**

Run:

```bash
git diff --check 288caa9..HEAD -- \
  docs/part1-system-basics/01-introduction.md \
  docs/part1-system-basics/02-ros2-architecture.md \
  tools/lecture_infographics/test_chapter_review_fixes.py
git status --short
git log -5 --oneline --decorate
```

Expected: 三个实施文件没有空白错误；状态中只保留用户原有的未跟踪 DOCX 和下载临时文件；最近提交清楚区分设计、第一讲修复和第二讲修复。

- [ ] **Step 3: 人工复核关键段落**

逐项确认：

1. 定义没有把开放世界作为前提；
2. 数据来源不再二分；
3. 比例控制有明确适用边界；
4. 人形适用性没有绝对比较；
5. 图号首次出现严格递增；
6. 九项资源都有官方链接、维护方和版本说明；
7. Episode 同时具有 metadata 与 steps；
8. 审核 DOCX、PNG 和其他章节没有进入提交。
