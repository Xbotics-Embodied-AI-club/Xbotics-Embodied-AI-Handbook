# Task 3 执行报告：美化七张 A 类概念示意图

## 结果

已完成 `1-1`、`1-2`、`1-4`、`1-5`、`1-6`、`1-8`、`2-1` 七张 A 类图的显式布局与成图。七张图统一使用共享标题区、结论条、语义配色和 3D 机器人素材；文字与箭头均由 Pillow 确定性绘制。`manifest.py` 未修改，B/C 类成图、第二讲 Markdown 和 `.crdownload` 均未保留任何变更。

## TDD：RED / GREEN

- RED：先在 `LayoutTests` 增加 `test_all_a_figures_have_explicit_layouts`，指定命令失败，原因为 `tools.lecture_infographics.layouts_a` 不存在，错误为 `ModuleNotFoundError`。
- GREEN：创建 `layouts_a.py`，`A_LAYOUTS` 精确包含七个 A 类键，并将 `render.py` 的 A 类调度改为 `render_a(...)`。
- 布局与渲染定向验证：3/3 通过。
- 最终完整测试：14/14 通过，用时 2.652 s。
- 额外产物检查：七张 A 图均为 1920×1080、RGB/RGBA，且包含 sRGB ICC profile；`git diff --check` 通过。

## 七张图的具体改进

- 图 1-1：以「数字 AI 屏幕」与「桌面抓取场景」左右对照，用蓝色行动箭头和绿色反馈箭头形成闭环。
- 图 1-2：以带相机机械臂为视觉中心，七个组成部分使用不同语义图标环绕，并用编号稳定阅读顺序。
- 图 1-4：保留五级技术路线阶梯，将「可靠执行—学习—优化—泛化—预测」作为能力扩展主线，并加入统一机械臂锚点。
- 图 1-5：统一视觉、语言、状态输入图标，用中央 VLA 融合核与 3D 机械臂串起数据流，并以黄/绿分层明确「动作建议」与「控制安全」责任边界。
- 图 1-6：左侧使用完整桌面场景，右侧三个候选动作统一机械臂视角；红色风险与绿色推荐结果一眼可辨。
- 图 1-8：五个场景按难度阶梯上升，分别配置桌面、移动、工业、家庭和人形机器人素材，底部趋势线与指标文字分离。
- 图 2-1：以带相机机械臂为硬件系统中心，六类硬件使用一致卡片、语义图标和指引线，避免原图的平均分散感。

## 视觉 QA

已生成七图联系表，并按原尺寸检查图 1-2、1-6、2-1；额外放大检查图 1-8。首轮发现图 1-8 指标文字压线、图 1-6 「推荐」过贴底边，修正后重新渲染并复查。最终未见标签遮挡、引导线交叉、中心空洞或机器人材质明显不一致。

## 变更文件

- 新增：`tools/lecture_infographics/layouts_a.py`
- 修改：`tools/lecture_infographics/render.py`
- 修改：`tools/lecture_infographics/test_render.py`
- 修改：七张 A 类 PNG 成图
- 新增：`.superpowers/sdd/task-3-report.md`

## 提交

提交信息：`美化第一二讲概念示意图`。最终提交哈希以任务回传状态为准。

## 自检与关注点

- 自检：`A_LAYOUTS` 键集与 manifest 的 A 类键集完全一致；渲染器在资产目录缺失时会回退到包内统一资产目录，保持测试可重复。
- 初版关注点（已在审查修复中解决）：图 1-8 的「家庭服务」曾复用移动操作平台素材；现已替换为独立客厅矢量场景。
- 非本任务的 `.superpowers/sdd/` 简报与既有未跟踪文件不纳入提交，仅显式添加本报告。

## 审查修复（2026-07-22）

审查修复提交：`f3be289` (`修复概念图审查问题`)。

### 追加 TDD 证据

- RED：新增 `test_household_stage_has_dedicated_scene_cues` 后，因 `HOUSEHOLD_SCENE_CUES` 不存在而报 `ImportError`；新增 `test_render_module_has_no_legacy_a_layouts` 后，因 `render.py` 仍暴露旧 A 实现而断言失败。
- GREEN：家庭场景约束和 A 单一来源约束通过；`LayoutTests + RenderTests` 5/5 通过。
- 完整回归：`python3 -m unittest tools.lecture_infographics.test_render -v` 为 16/16 通过，用时 2.593 s。
- 静态边界：`render.py` 中已无 `visual_1_*`、`visual_2_1`、`generic_a` 或旧 `2-1` 特判；`git diff --check` 通过。

### 追加修复内容

- Important 1：图 1-8 「家庭服务」不再复用移动操作素材，改为窗户、沙发、落地灯和茶几组成的独立客厅矢量场景，与第 2 阶形成明确区分。
- Important 2：删除 `render.py` 中七张旧 A 函数、`generic_a`、旧 A custom dispatch 与仅为旧 A 服务的辅助绘图函数；A 类现只由 `layouts_a.py` 提供。
- Minor：图 1-2 和 2-1 的中央 3D 素材外增加有意的白色圆角承载卡和柔和描边，消除白色矩形底像贴图截断的观感。
- 保留项：资产目录缺失时的包内静默回退未修改，以继续支持现有临时目录渲染测试。

### 追加视觉 QA

重新生成七张 A 图后，按原尺寸复查图 1-8、1-2、2-1。图 1-8 的家庭环境语义清楚，与移动机器人无视觉混淆；图 1-2 和 2-1 的中央承载卡边界完整、留白均匀，未造成标签或指引线遮挡。
