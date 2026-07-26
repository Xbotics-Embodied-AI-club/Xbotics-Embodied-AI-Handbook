# 第五讲图片来源与制作记录

核验日期：2026-07-25

本目录分为两类：

- `original/`：为本讲制作的原创插画和教学示意图。
- `ref/`：从具有明确来源与许可证的官方仓库取得的素材。

## 原创半写实插画

### `original/real-vs-sim-hero.png`

- 用途：图 5-1，真实硬件试错与仿真实验的对照。
- 制作方式：使用 OpenAI ImageGen 生成；无输入参考图。
- 提示词：横向教材插画，左侧是配有急停按钮的 SO-101 实验台，研究者谨慎观察真实机械臂；右侧是同一机械臂的仿真数字孪生，同时重复执行多个方块实验。蓝白技术教材配色，半写实 3D 插画，构图克制，留有图注空间，不出现文字、标志和水印。
- 修改：生成结果仅按本章文件命名保存，未做内容编辑。
- SHA-256：`f63cc0ce11ad6f43c69166874874ad413923a6bcdc5ad71e2d7fbf1fd7a28f99`

### `original/manipulation-vs-locomotion.png`

- 用途：图 5-2，SO-101 主实验与 G1 迁移验证。
- 制作方式：使用 OpenAI ImageGen 生成；无输入参考图。
- 提示词：横向教材插画，左侧为 SO-101 完成 reach 与方块 pick-place，右侧为 Unitree G1 在平地速度跟踪场景中行走；两侧由同一条蓝色实验主线连接，表达 manipulation 与 locomotion 共用任务契约。蓝白教学风，半写实 3D，灰度打印仍能辨识，不出现文字、标志和水印。
- 修改：生成结果仅按本章文件命名保存，未做内容编辑。
- SHA-256：`2cb72cf3c6d37ac5e6fb45034ce9dccb7b39f0c252aad1687727c4d8a67f61ae`

## 原创矢量教学图

以下图片由 `generate_figures.cjs` 生成，布局、文字和图形均为本项目原创；脚本同时输出 SVG 与 PNG。所有图均采用蓝白配色，并用形状、编号、线型和位置传达关键信息，避免只靠颜色区分。

| 文件前缀 | 内容 |
|---|---|
| `05-03-task-contract` | 七项任务契约 |
| `05-04-contract-implementation-map` | 契约到 MJCF / Python 的映射 |
| `05-06-reach-control-loop` | SO-101 reach 闭环 |
| `05-07-pick-place-failures` | pick-place 四类失败 |
| `05-08-simulation-clocks` | 物理、控制、策略与渲染四个时钟 |
| `05-09-isaaclab-environment-flow` | Isaac Lab 场景层与任务层 |
| `05-10-parallel-evaluation` | 单次 Demo 与条件分布评测 |
| `05-12-g1-fall-timeline` | G1 摔倒时间序列诊断 |
| `05-13-sim2real-gaps` | Sim2Real gap 与契约项的对应关系 |
| `05-14-experiment-loop` | 单变量实验和失败复盘闭环 |
| `05-15-platform-choice` | 任务驱动的平台选择 |

## 原创 ImageGen 教学插画（2026-07-25 版）

以下图片位于 `original/imagegen/`，使用 OpenAI ImageGen 生成或定点编辑。它们以原矢量图的教学命题和正文论述为信息约束，并在需要保持机器人身份时参考本目录已登记的 SO-101、Franka 与 G1 官方图片。生成方向统一为：横向 16:9、书籍级科学插画、柔和半写实 3D 与精确分析标注结合、米白底、钴蓝/青绿主色、少量橙色告警；用空间、轨迹、器材和状态变化组织信息，避免僵硬卡片网格；不生成水印或虚构产品界面。旧版 SVG/PNG 均保留，便于回滚与编辑。

| 文件 | 用途与提示词摘要 | 定点修改 | SHA-256 |
|---|---|---|---|
| `05-03-task-contract-imagegen.png` | 图 5-3：SO-101 实验台居中，七项任务契约环绕，强调“任务是一个可复盘实验系统”。 | 依据官方 SO-101 图片校正机械臂外形。 | `3533b5327b90793306f624d2733caf4dd13b836c93ffa17d54a7da723246354c` |
| `05-05-contract-implementation-imagegen.png` | 图 5-5：以 SO-101 为桥梁，左侧展开 MJCF 的 body/joint/geom/actuator/timestep，右侧呈现 Python 闭环变量。 | 将伪代码屏幕收敛为 `target/qpos/ctrl/step/log` 五个关键字。 | `28293a0dbaf6ebf4a5ba77a09ebf63f7d820021cdc10ca7afc471e776696c9f2` |
| `05-06-reach-control-loop-imagegen.png` | 图 5-6：SO-101 多重残影表现 reach 运动，环形链路串起 target/error/Jacobian/IK/command/control/physics/state，并标明 5 ms 与 20 ms。 | 移除生成的品牌贴纸，保持纯器材表面。 | `6a0adcc8784f57e9aa6979e0cbcdab0aedfc8abc78132e67c4f1253dab03936b` |
| `05-07-pick-place-failures-imagegen.png` | 图 5-7：用连续实验台场景表现抓空、夹持滑落、加速度甩出和放置滑移四种不同失败机制。 | 无。 | `e5701204800ce566a025917b682fd94d22277dca357d24a7fd607fd8e3697cb4` |
| `05-08-simulation-clocks-imagegen.png` | 图 5-8：机械时钟与齿轮时间轴表现 physics/control/policy/render 四类时钟，突出 4×5 ms=20 ms 的嵌套关系。 | 无。 | `a0a1b28fa97305f2ed5237e9047b302ef7cf071f2ca687a10bbe2d311c72c1a9` |
| `05-10-isaaclab-environment-flow-imagegen.png` | 图 5-10：Franka 场景剖面与环形数据流，串起 observation/action/command/reward/termination/metrics，并把 scene/articulation/sensor 组织成环境。 | 校正 `ARTICULATION` 术语拼写。 | `46b70f3a2cd055bb2394f5cc353df4e09980e30194f752c352859bfe989ec539` |
| `05-11-parallel-evaluation-imagegen.png` | 图 5-11：从单次 Franka 运行展开为批量环境阵列，配合成功率、误差分布及 target/pose/physics variation。 | 将深色仪表盘改为适合书籍印刷的浅色背景。 | `d8ce89ce9485a55bd00de5fe66150e9842014ef273315f5589be3c5ee0b56866` |
| `05-11-parallel-evaluation-imagegen-v3.png` | 图 5-11（第五讲 v3）：保留单次运行、批量环境、成功比例和误差分布的概念关系，但不呈现任何实测数值。 | 将未经实验支持的 `73%` 改为定义式 `S / N`；其余构图保持不变。 | `82e4ff3ec5fde77b7d4709ce12d4bd5145d6176faeb561b4e99e1cd499766743` |
| `05-12-platform-choice-imagegen.png` | 图 5-12：研究者从“我要观察什么”出发，面向模型控制、并行评测、ROS 联调和视觉领域四个实验世界选择工具。 | 无。 | `3aa1400515f8f4780e81bf5fe5e9c4f9246b9b8da9c68a8aeaddd966d15dfd29` |
| `05-15-g1-fall-timeline-imagegen.png` | 图 5-15：G1 由稳定行走到摔倒的残影序列，与 command、velocity、foot contact、roll 四条同步曲线对齐，突出 first anomaly。 | 无。 | `91106f880d742d6150ac147786a88cb5a2bf077d2b2cbc07638b147ff3587e46` |
| `05-16-sim2real-gaps-imagegen.png` | 图 5-16：仿真 SO-101 与真实 SO-101 隔台相望，六条差距通道对应 model/contact/sensing/actuator/latency/vision。 | 无。 | `9ad22fa7f5398907e777e4bf6f84130305c8de16a44561e91dc473fdf4e01b41` |
| `05-18-experiment-loop-imagegen.png` | 图 5-18：以 SO-101 实验为中心，用真实实验器材和手写记录串起 baseline/one variable/predict/measure/evidence/next 六步闭环。 | 移除所有品牌、型号与吉祥物贴纸。 | `9b68885b4ee99db7a32aefefe6e576d638eb37aabba9c206c2f903eb30803cf6` |

## 官方仓库素材

### `ref/so101-follower-official.webp`

- 用途：SO-101 follower 实物结构。
- 作者/组织：TheRobotStudio。
- 仓库：[`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100)
- 原始文件：[`media/SO101_Follower.webp`](https://github.com/TheRobotStudio/SO-ARM100/blob/fda892cba81032c46c40976a48c9ceadbf40a9ca/media/SO101_Follower.webp)
- 固定提交：`fda892cba81032c46c40976a48c9ceadbf40a9ca`
- 许可证：Apache License 2.0（仓库许可证；未发现该媒体文件的单独许可证）。
- 修改：无。
- SHA-256：`855809851ecf2ac5a28b2f0050b4baca3adc5a18c5175908399f9c6a52dd6877`

### `ref/so101-menagerie.png`

- 用途：本讲 MuJoCo 实验采用的 SO-101 仿真资产。
- 作者/组织：Google DeepMind MuJoCo Menagerie，模型源自 TheRobotStudio。
- 仓库：[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
- 原始文件：[`robotstudio_so101/so101.png`](https://github.com/google-deepmind/mujoco_menagerie/blob/71f066ad0be9cd271f7ed58c030243ef157af9f4/robotstudio_so101/so101.png)
- 固定提交：`71f066ad0be9cd271f7ed58c030243ef157af9f4`
- 许可证：Apache License 2.0（模型目录内 `LICENSE`）。
- 修改：无。
- SHA-256：`fca552381f318dab5b1f9a69007e79d905d91c9d1bebbfdf267b78d924a9c46d`

### `ref/unitree-g1-menagerie.png`

- 用途：G1 机器人资产与结构说明。
- 作者/组织：Unitree Robotics，经 MuJoCo Menagerie 发布。
- 仓库：[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
- 原始文件：[`unitree_g1/g1.png`](https://github.com/google-deepmind/mujoco_menagerie/blob/71f066ad0be9cd271f7ed58c030243ef157af9f4/unitree_g1/g1.png)
- 固定提交：`71f066ad0be9cd271f7ed58c030243ef157af9f4`
- 许可证：BSD 3-Clause（模型目录内 `LICENSE`）。
- 修改：无。
- SHA-256：`72bb2dd0bf7fab732229554d1bd80b561aa161a9dabe45778ffa70193e05e1e7`

### `ref/isaaclab-franka-reach.jpg`

- 用途：Isaac Lab 官方 Franka reach 并行环境。
- 作者/组织：NVIDIA / Isaac Lab contributors。
- 仓库：[`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab)
- 原始文件：[`docs/source/_static/tasks/manipulation/franka_reach.jpg`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/docs/source/_static/tasks/manipulation/franka_reach.jpg)
- 固定标签/提交：`v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0`
- 许可证：BSD 3-Clause（Isaac Lab 仓库许可证）。
- 修改：无。
- SHA-256：`a2dc7ce3445fc8e1a6f8fd462c5f7b61acd55131fe3907585f7c6d7eea8a6745`

### `ref/isaaclab-g1-flat.png`

- 用途：Isaac Lab 官方 G1 平地速度跟踪环境。
- 作者/组织：NVIDIA / Isaac Lab contributors；G1 资产版权归 Unitree Robotics。
- 仓库：[`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab)
- 原始文件：[`docs/source/_static/tasks/locomotion/g1_flat.jpg`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/docs/source/_static/tasks/locomotion/g1_flat.jpg)
- 固定标签/提交：`v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0`
- 许可证：Isaac Lab BSD 3-Clause；G1 资产另见仓库内 `docs/licenses/assets/unitree-license.txt`（BSD 3-Clause）。
- 修改：原文件虽然使用 `.jpg` 后缀，实际内容为 PNG；本地仅纠正扩展名，像素内容未修改。
- SHA-256：`9b934872a68d4d0b7a548d7056e4185751dffa13e8e7dd1d3c14fd2089c09df0`

### `ref/isaaclab-g1-rough.png`

- 用途：Isaac Lab 官方 G1 崎岖地形速度跟踪环境。
- 作者/组织：NVIDIA / Isaac Lab contributors；G1 资产版权归 Unitree Robotics。
- 仓库：[`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab)
- 原始文件：[`docs/source/_static/tasks/locomotion/g1_rough.jpg`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/docs/source/_static/tasks/locomotion/g1_rough.jpg)
- 固定标签/提交：`v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0`
- 许可证：Isaac Lab BSD 3-Clause；G1 资产另见仓库内 `docs/licenses/assets/unitree-license.txt`（BSD 3-Clause）。
- 修改：原文件虽然使用 `.jpg` 后缀，实际内容为 PNG；本地仅纠正扩展名，像素内容未修改。
- SHA-256：`69ee9c49e1d84b41cb78527e0a538517bbc9240f692bb5a286b6f078e5728283`

## 出版使用提示

- 每张外部图片的正文图注都应给出组织、仓库/标签和许可证；本清单保留完整 URL 与校验值。
- Apache-2.0 和 BSD-3-Clause 的再发布条件仍需在整书的版权页或第三方许可附录中统一履行。
- 文件名、品牌和机器人型号只用于识别来源，不表示权利方对本书的背书。
