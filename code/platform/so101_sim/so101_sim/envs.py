"""KIT 双相机任务 —— 用我们真机的 SO101（KIT 版）+ top/wrist 两路相机重建 task1 视角。

与 vendored squint 的单 base_camera 环境不同，这里的观测有**两路相机**，对齐真机数据集
`task1`（observation.images.top / observation.images.wrist，均 480×640）：

- **top**：绑在 URDF 的 `top_camera_optical_frame`（挂在 base_link 上，随支架固定俯视全局）。
- **wrist**：绑在 `wrist_camera_optical_frame`（挂在 gripper_link 上，随夹爪动、看被抓物）。

两个光学系的位姿已由真机标定写进 KIT URDF，无需 look_at 手调。唯一要补的是 **ROS 光学系
（z 前 / x 右 / y 下）→ SAPIEN 相机系（x 前 / y 左 / z 上）** 的固定旋转，写成常量四元数
`_ROS_OPTICAL_TO_SAPIEN` 挂在相机的相对位姿上。

任务语义沿用 tasks/place.py 的 Place（放入料盒才算成功）：本文件只换机器人 + 相机，
不改任务逻辑。so101_kit 与 vendored so101 同构（同 6 个活动关节），故基座朝向与 rest qpos
沿用 so101。
"""

import re
from pathlib import Path

import numpy as np
import sapien
import torch
import trimesh
from sapien.render import RenderBodyComponent, RenderMaterial, RenderShapeTriangleMesh
from transforms3d.euler import euler2mat
from transforms3d.quaternions import qmult, quat2mat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

from so101_sim.tasks.place import Place
from so101_sim.robots.so101_base.so101 import SO101

from so101_sim.robots.so101_kit_slow import ITEM_FRICTION_RANGE, SIM_FPS

# 真机 KIT 演示套件的物体 mesh（由 assets/objects 的 STEP 用 convert_step.py 转出）。
_OBJECTS_DIR = Path(__file__).resolve().parent / "robots" / "kit_assets" / "objects"

# ROS 光学系 → SAPIEN 相机系的固定旋转（wxyz）：
# 相机前(x)=光学前(z)、相机左(y)=光学左(-x)、相机上(z)=光学上(-y)。
_ROS_OPTICAL_TO_SAPIEN = [0.5, 0.5, -0.5, 0.5]
# 真机相机是**正装**的（与标定一致）；仿真这条光学系约定给反了，直接渲出来的 top/wrist
# 会相对真机上下颠倒。此处绕相机光轴（SAPIEN 相机前向 x 轴）补一个 180° roll，让**仿真
# 朝实物靠拢**——即渲出与正装真机 + 标定一致的画面，而不是在图像上做后处理翻转。
_ROLL_180 = [0.0, 1.0, 0.0, 0.0]  # 绕 x 轴 180°（wxyz）
_OPTICAL_CONV = list(qmult(_ROS_OPTICAL_TO_SAPIEN, _ROLL_180))

# 相机分辨率对齐真机 task1（宽 640 × 高 480）。
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 竖直 FOV（弧度）= ChArUco 棋盘格标定结果 fovy=59.17°（RMS 0.72px，233 帧，640×480）。
# 标定视频/部署/task1 是同一路相机（同型号同镜头），故 top 与 wrist 共用此值。
# CameraConfig.fov 即 SAPIEN fovy，标定值可直接用。
CAMERA_FOV = np.deg2rad(59.17)
TOP_CAMERA_FOV = CAMERA_FOV
WRIST_CAMERA_FOV = CAMERA_FOV

# 场景外观对齐真机布置：黑色机械臂 + 黑色相机支架、白色台面。
# 机械臂与支架都是 KIT URDF 里的 link，统一走 squint 的 robot_color 旋钮刷黑；
# 其余物体（方块/圆柱/料箱）的颜色已编码进各自资产文件名，此处不动。
ROBOT_COLOR = (0.05, 0.05, 0.05)
TABLE_COLOR = (0.9, 0.9, 0.9, 1.0)

# top 相机相对挂载光学系的位置微调（米，光学系 x 右 / y 下 / z 前）。标定视角与真机对齐后
# 定死；(0,0,0) 表示完全用 URDF 里标定的挂载位姿。
TOP_CAMERA_OFFSET = (0.0, 0.0, 0.0)
# top 相机额外俯仰（度，绕光学系 x 轴=画面水平轴；正=镜头更朝下）。
# URDF 的挂载位姿是 KIT 设计值；真机 task1 那套 rig 的实际装法与设计值有差，
# 而标定板只能标 wrist（eye-in-hand）、标不了 top（eye-to-hand），只能靠画面反解。
# 0 = 完全用 URDF 设计位姿。
TOP_CAMERA_EXTRA_PITCH_DEG = 0.0


def _paint_actor(actor, rgba):
    """把一个 actor 的所有渲染面刷成纯色 rgba，并清掉原有贴图（如木纹）。"""
    for obj in actor._objs:
        entity = obj.entity if hasattr(obj, "entity") else obj
        render_body = entity.find_component_by_type(RenderBodyComponent)
        if render_body is None:
            continue
        for render_shape in render_body.render_shapes:
            for part in render_shape.parts:
                mat = part.material
                mat.set_base_color(list(rgba))
                # 清掉所有贴图，否则木纹的漫反射/法线/粗糙度会留在表面、白不彻底。
                for tex_attr in ("diffuse_texture", "normal_texture",
                                 "roughness_texture", "metallic_texture"):
                    if hasattr(mat, tex_attr):
                        setattr(mat, tex_attr, None)
                # 提高粗糙度、压低金属度，得到均匀的哑光白（不反光、无高光纹路）。
                mat.roughness = 1.0
                mat.metallic = 0.0


def _mesh_facts(glb_path):
    """读一个 glb 的包围盒全尺寸(x,y,z 米)、几何中心、以及顶点色（来自 STEP 文件名的 #RRGGBB）。"""
    m = trimesh.load(str(glb_path), force="mesh")
    rgba = np.asarray(m.visual.main_color, float) / 255.0
    return (np.asarray(m.bounding_box.extents, float),
            np.asarray(m.bounding_box.centroid, float),
            rgba)


def mesh_full_size(mesh_name):
    """物体 mesh 的真实全尺寸（米），取自 STEP 转出的 glb 包围盒。"""
    return _mesh_facts(_OBJECTS_DIR / f"{mesh_name}_visual.glb")[0]


# ── STEP 文件名把两个方块的颜色标反了：按真机视频与物料表校正 ────────────────
#
# 颜色的传递链是 STEP 文件名 `#RRGGBB` → glb 顶点色 → `base_color`，转换本身没问题。
# 错在源头：`cube_4_#b74e4d.STEP`（红）/ `cube_2_#9cbbd1.STEP`（淡蓝），而
#   · 用户物料表：海绵方块 **2cm 红 / 4cm 蓝**，海绵圆柱 φ4×4cm 蓝
#   · 真机视频（ModelScope `pick_up_a_cube_and_place_in_the_bin`，top 相机）：
#     被抓的方块量得 **3.35–4.78cm**（以料箱 8×10cm 定标）= 4cm 那个，
#     像素平均 **RGB[122,145,157]**，与 `#9cbbd1`(156,187,209) 同色系，绝非 `#b74e4d`(183,78,77)。
# 两个独立来源一致，故判 STEP 文件名标反，此处按真值覆盖（不改 STEP、不改 glb，只改渲染色）。
# 圆柱的 `#9cbbd1` 与物料表一致，不动。
#
# 这不是"好看不好看"的问题：数据要拿去训 VLA，指令说「拿蓝色方块」而画面是红的，
# 语言接地直接错位。
MESH_COLOR_OVERRIDE = {
    "cube_4": (0.612, 0.733, 0.820),      # #9cbbd1 淡蓝
    "cube_2": (0.718, 0.306, 0.302),      # #b74e4d 红
}


def _swap_visual_to_mesh(actor, glb_path, target_full_sizes, align="center",
                         mesh_rotation=None):
    """把一个 Actor 每个并行子实例的渲染网格换成 glb（保留原碰撞盒，抓取判定/reward 不变）。

    - `target_full_sizes`：形如 (num_envs, 3) 的目标全尺寸（米），把 mesh 各轴缩放到与原碰撞盒一致。
    - `align`："center" 让 mesh 包围盒中心对到 actor 原点（方块/圆柱）；"bottom" 让 mesh 底面贴 z=0（料箱）。
    在 `_load_scene` 里调用（GPU 烘焙前），换出的网格能进 GPU 渲染——与 `_paint_actor` 同一时机。

    **颜色必须显式给 RenderMaterial**：`RenderShapeTriangleMesh(filename=...)` 不会采用 glb 里的
    顶点色，缺 material 时渲出来是灰白件（实测 item 像素中位色 178,178,178 而非 STEP 的 183,78,77）。
    这里把 glb 顶点色（即 STEP 文件名 `#RRGGBB`）取出来喂 `base_color`，渲染才是真机件的颜色。

    **`mesh_rotation`（wxyz 四元数）修 mesh 建模轴与物理体轴不一致**：STEP 各件的建模轴不统一
    —— `cube_4` 沿 z（包围盒中心 z=2cm），而 `cylinder_4` **沿 y**（中心 y=2cm）。squint 的圆柱
    碰撞体是绕 y 转 90° 立起来沿 z 的，只做平移不做旋转 ⇒ **视觉圆柱躺着、物理圆柱立着**
    （实测：腕部视角里圆面朝相机）。方块对称所以不受影响。旋转在缩放/对齐之前作用于 mesh 局部系。
    """
    ext, center, rgba = _mesh_facts(glb_path)
    override = MESH_COLOR_OVERRIDE.get(Path(glb_path).stem.replace("_visual", ""))
    if override is not None:
        rgba = np.array([*override, 1.0], float)
    ext = np.where(ext < 1e-6, 1.0, ext)  # 防 0 除（薄面）
    if mesh_rotation is not None:
        # 旋转会把包围盒的轴对调，缩放与对齐都要按旋转后的量算
        rot = np.asarray(quat2mat(mesh_rotation), float)
        ext = np.abs(rot @ ext)
        ext = np.where(ext < 1e-6, 1.0, ext)
        center = rot @ center
    for i, obj in enumerate(actor._objs):
        entity = obj.entity if hasattr(obj, "entity") else obj
        old = entity.find_component_by_type(RenderBodyComponent)
        if old is not None:
            entity.remove_component(old)
        full = np.asarray(target_full_sizes[i], float)
        scale = (full / ext).astype(np.float32)
        material = RenderMaterial(base_color=rgba.tolist(), roughness=0.7, metallic=0.0)
        shape = RenderShapeTriangleMesh(filename=str(glb_path), scale=scale, material=material)
        # 缩放后 mesh 中心/底面在局部系的落点，平移回来对齐 actor 原点。
        c = center * scale
        q = list(mesh_rotation) if mesh_rotation is not None else [1.0, 0.0, 0.0, 0.0]
        if align == "bottom":
            shape.set_local_pose(sapien.Pose(p=[-c[0], -c[1], (full[2] / 2) - c[2]], q=q))
        else:
            shape.set_local_pose(sapien.Pose(p=[-c[0], -c[1], -c[2]], q=q))
        rb = RenderBodyComponent()
        rb.attach(shape)
        entity.add_component(rb)


# ── KIT 与 vanilla 的 wrist_roll 零位差 90°：起始位姿必须换算 ──────────────────
#
# ★这是一处真实资产 bug，2026-08-14 才查出来。squint 任务把 `rest_qpos` 设成
# **vanilla** `SO101.keyframes["start"]`，其 wrist_roll = −π/2 = −90°。而 KIT URDF 的
# wrist_roll 限位是 **[−67.21°, 252.79°]** —— **−90° 在界外 22.79°**。
#
# 证据（两份 URDF 的 wrist_roll 行程宽度都恰好 320.00°，整体平移 +90.00°）：
#     vanilla [−157.21, 162.79]   KIT [−67.21, 252.79]
# ⇒ `KIT_wrist_roll = vanilla_wrist_roll + 90°`，vanilla 的 −90° 在 KIT 下应是 **0°**。
# 真机 home 的 wrist_roll = −0.53° 也落在 0 附近，与该换算自洽。
#
# 不修的后果远不止"起始跳一下"（实测每集第 1 步 wrist_roll 硬跳 23.699°，之后 200 步恒定
# —— 这是**限位截断**，不是重力下垂：下垂会随时间收敛且分布在多个关节）：
# **控制器 target 被永久钉在限位外 22.8°**，delta 控制器的一切增量都从这个够不到的基准累加，
# 在策略把它命令回 −67.21° 以上之前该关节完全不动 = 腕部滚转有 22.8° 死区，
# 且整个抓取搬运过程腕部都贴着限位。此前长期查不出的"擦棱夹持而非两侧夹住"、
# "给定固定姿态的 IK 大面积无解"，都能由此解释：腕部姿态自由度实际被限位吃掉了。
KIT_WRIST_ROLL_OFFSET = np.pi / 2


def kit_rest_qpos():
    """vanilla `start` keyframe 换算到 KIT 关节约定后的起始位姿。"""
    q = np.asarray(SO101.keyframes["start"].qpos, float).copy()
    q[4] += KIT_WRIST_ROLL_OFFSET          # 索引 4 = wrist_roll
    return q.tolist()


class KitDualCameraMixin:
    """把观测相机换成绑在 KIT URDF 光学系上的 top + wrist 两路，并对齐真机的黑臂白台外观。

    可选：设 `ITEM_MESH` / `BIN_MESH` 为 kit_assets/objects 下的物体名，则把 squint 内置的
    盒/柱 item 与料箱的**渲染网格**换成真机演示套件的 STEP→mesh（碰撞盒不动，抓取/reward 不变）。
    """

    SUPPORTED_ROBOTS = ["so100", "so101", "so101_kit"]

    # 子类可覆盖：把 item / bin 的视觉换成真机套件 mesh（None=沿用 squint 内置几何）。
    ITEM_MESH = None
    BIN_MESH = None
    # mesh 建模轴 → 物理体轴的修正四元数（wxyz）。None=两者一致，不用转。
    # 各 STEP 件建模轴不统一，见 `_swap_visual_to_mesh` 的说明。
    ITEM_MESH_ROTATION = None
    BIN_MESH_ROTATION = None

    def __init__(self, *args, robot_uids="so101_kit", domain_randomization_config=None, **kwargs):
        # 基座朝向与 so101 一致（z 轴不旋转）；起始 qpos 必须做 wrist_roll 零位换算，
        # 否则 −90° 落在 KIT 限位外，见 `kit_rest_qpos` 上方的说明。
        self.base_z_rot = 0
        self.rest_qpos = kit_rest_qpos()

        # 把机械臂+相机支架刷黑：注入 robot_color 到任务的 DR 配置里，由 squint 的
        # _randomize_robot_color 统一着色（KIT URDF 里支架就是机器人的 link）。
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        config.setdefault("robot_color", list(ROBOT_COLOR))

        super().__init__(*args, robot_uids=robot_uids, domain_randomization_config=config, **kwargs)

    def _load_scene(self, options: dict):
        # 先按任务原逻辑搭好桌子/物体/机器人着色，再把 mani_skill 自带的木桌面刷白。
        super()._load_scene(options)
        _paint_actor(self.table_scene.table, TABLE_COLOR)

        # 把 item / bin 的渲染网格换成真机演示套件的 STEP→mesh（碰撞盒保持不变）。
        # item_dimensions / bin_dimensions 都是每 env 的半尺寸，×2 得全尺寸喂给缩放。
        if self.ITEM_MESH is not None:
            item_full = self.item_dimensions.cpu().numpy() * 2
            _swap_visual_to_mesh(self.item, _OBJECTS_DIR / f"{self.ITEM_MESH}_visual.glb",
                                 item_full, align="center",
                                 mesh_rotation=self.ITEM_MESH_ROTATION)
        if self.BIN_MESH is not None and hasattr(self, "bin"):
            bin_full = self.bin_dimensions.cpu().numpy() * 2
            _swap_visual_to_mesh(self.bin, _OBJECTS_DIR / f"{self.BIN_MESH}_visual.glb",
                                 bin_full, align="bottom",
                                 mesh_rotation=self.BIN_MESH_ROTATION)

    @property
    def _default_sensor_configs(self):
        conv = sapien.Pose(q=_OPTICAL_CONV)
        # top 相机在挂载光学系里额外平移 + 额外俯仰（对齐真机视角用），wrist 不动。
        # ★俯仰是绕 **y** 轴（实测：在 `_OPTICAL_CONV` 之后右乘绕 y 的旋转，+10° 恰好让
        # 俯视角 67.5°→77.5°；绕 x 无效果、绕 z 只产生 -2° 的耦合，都不是俯仰轴）。
        half = np.deg2rad(TOP_CAMERA_EXTRA_PITCH_DEG) / 2
        pitch_q = [np.cos(half), 0.0, np.sin(half), 0.0]
        top_pose = sapien.Pose(p=list(TOP_CAMERA_OFFSET),
                               q=list(qmult(_OPTICAL_CONV, pitch_q)))
        links = self.agent.robot.links_map
        return [
            CameraConfig(
                "top",
                pose=top_pose,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fov=TOP_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=links["top_camera_optical_frame"],
            ),
            CameraConfig(
                "wrist",
                pose=conv,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fov=WRIST_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=links["wrist_camera_optical_frame"],
            ),
        ]


# ── 真机套件物体的「真实尺寸」抓取任务 ────────────────────────────────────
#
# 上面那批 `SO101Kit*Cube-v1` 沿用 squint 默认碰撞尺寸（cube 半边长 2.2–2.8cm 档），
# 只换了视觉网格 —— 于是 4cm 的 cube_4 会被压缩渲染成 ~2.5cm，画面与真机件不同尺寸。
# 下面这批把**碰撞盒也钉到 STEP 实测真值**，视觉与物理同尺寸，mesh 缩放恒为 1.0：
#
#   cube_4      4.0 × 4.0 × 4.0 cm    #b74e4d
#   cube_2      2.0 × 2.0 × 2.0 cm    #9cbbd1
#   cylinder_4  直径 4.0 cm / 高 4.0 cm  #9cbbd1
#
# 每个物体注册**一对**任务：
#   `SO101Lift<Obj>Real-v1`    —— vanilla 外观 + 单 base_camera，给 RL 专家训练用
#                                （专家的 CNN 编码器是 3 通道，吃不了 KIT 的双相机 6 通道）。
#   `SO101KitLift<Obj>Real-v1` —— KIT 双相机 + mesh + STEP 颜色，给 replay 重渲数据集用。
# 两者碰撞尺寸逐字一致，所以 vanilla 里录的 env_states 灌进 KIT 是同一套物理。
#
# 尺寸的唯一真相源是 STEP→glb 的包围盒（`mesh_full_size`），不在代码里另抄一份数字。


# 真机演示套件的件是**海绵**（物料表：海绵方块 2cm 红 / 4cm 蓝，海绵圆柱 φ4×4cm 蓝），
# 不是 3D 打印实心件。取 200 kg/m³ ⇒ 4cm 件约 12.8g，与软质泡沫同量级。
#
# ★这条曾被改成 1250（按"PLA 实心 80g"），那是把材质臆断成 3D 打印件的结果，是资产错误。
# 它的连锁后果比数字本身大得多：件一重，指尖托运就撑不住，于是我判定"squint 原生奖励
# 学不会夹紧"，进而叠了一整套几何夹持奖励与成功门（v9–v14），把原本能到 success 1.00 的
# 配方压到 0.008。资产回真值后这些补丁全部失去前提，应一并退掉。
# 另注：200 恰是 squint 默认值 —— 它的奖励尺度/成功阈值都是围绕这个量级调的，用真值等于
# 回到配方的适用域内。
ITEM_DENSITY = 200.0


def _cube_size_config(mesh_name):
    """把 STEP 方块的真实边长与密度钉成 squint 的区间（上下界相同=不随机）。"""
    half = float(mesh_full_size(mesh_name)[0]) / 2
    return {"cube_half_size_range": (half, half),
            "item_density_range": (ITEM_DENSITY, ITEM_DENSITY)}


def _cylinder_size_config(mesh_name):
    """把 STEP 圆柱的真实直径/高与密度钉成 squint 的 can 区间。"""
    ext = mesh_full_size(mesh_name)
    half_radius = float(ext[0]) / 2
    half_height = float(ext[2]) / 2
    return {"can_radius_range": (half_radius, half_radius),
            "can_half_height_range": (half_height, half_height),
            "item_density_range": (ITEM_DENSITY, ITEM_DENSITY)}


# ★撒点区照抄真机分布，不照抄 IK 扫描。
# 依据：ModelScope cube 任务 174 集，用**夹爪开合**定位关键时刻再正解算末端位置
# （闭合那刻末端在物体上、张开那刻在料箱上，见 hybrid/infer_real_layout.py）：
#   物体 x p5..p95 = 0.172..0.368   y = -0.099..0.190
#   料箱 x         = 0.199..0.387   y = -0.064..0.179
#   两者水平间距 中位 13.7cm（p5 8.9 / p95 22.0）
# ⚠️ 我曾用 IK 网格扫描得出"x>=0.325 全不可达、撒点区一半超臂展"，据此把区域收到
# 10×10cm —— **该结论已作废**：真机确实够到 x=0.368，且我们此前 replay 过真机轨迹并
# 目检通过，说明这些位姿在仿真里可达。是我给 IK 强加的末端姿态约束太严造成的假阴性。
# 教训：可达性要用**真机既有轨迹**证伪，不能只信自己写的求解器。
# 真机 y 明显偏正（中位 +0.042~+0.053），故中心不取 y=0。
SPAWN_BOX_POS = [0.27, 0.045]
SPAWN_BOX_HALF_SIZE = 0.10


class ReachableSpawnMixin:
    """把物体/料箱的撒点区对齐真机实测分布。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("spawn_box_pos", list(SPAWN_BOX_POS))
        kwargs.setdefault("spawn_box_half_size", SPAWN_BOX_HALF_SIZE)
        super().__init__(*args, **kwargs)


class RealSizeItemMixin:
    """把 item 的碰撞尺寸钉到 `ITEM_MESH` 对应 STEP 件的真实尺寸。

    squint 的 Lift 用 `domain_randomization_config` 的尺寸区间造碰撞体，这里把区间上下界
    都设成 STEP 真值即可——不改 vendored 任务逻辑，reward/success 判定原样沿用。
    """

    ITEM_SIZE_CONFIG = {}

    def __init__(self, *args, domain_randomization_config=None, **kwargs):
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        for key, value in self.ITEM_SIZE_CONFIG.items():
            config.setdefault(key, value)
        super().__init__(*args, domain_randomization_config=config, **kwargs)


# `cylinder_4` 的 STEP 建模轴沿 y（包围盒中心 y=2cm），而 squint 的圆柱碰撞体立着沿 z
# ⇒ 视觉必须绕 x 转 -90° 把 y 轴掰到 z 轴，否则渲出来是躺着的圆柱（实测过）。
_CYLINDER_Y_TO_Z = [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0]


# ── 真机速度包线版任务（`*Slow-v1`）──────────────────────────────────────
#
# 上面那批任务用 squint 默认的 delta 上限（臂 ±0.1 rad/step），20fps 下饱和速度 114.6°/s，
# 而用户真机 task1 实测 p95 只有 29–66°/s ⇒ 生成的动作快 1.6–3.6×，真机跟不上。
# 这批把动作空间压进真机包线（逐关节按 `真机 p95 / fps` 反算，见 `robots/so101_kit_slow.py`），
# 重训专家后生成的轨迹才是"真机能执行的速度下的真解"。
#
# 命名 `*Slow-v1`；与上面那批**并存不替换**，便于对照与回退。


# ── 「真夹住」的判据：几何贴合，不是接触力 ────────────────────────────
#
# ★为什么不能用接触力：SAPIEN/PhysX 的 `contact_offset = 0.02` —— 两体相距 **2cm 以内**
# 就生成接触约束并报接触力。所以"两指各 16–18N"与"画面里方块离爪明显有缝"可以同真，
# **任何基于力的判据在这个设定下天然分不清「夹住」和「靠近」**。
# 实测（用两爪碰撞网格全部顶点到方块盒的最小距离，不经任何代表点）：
#     gripper_link 中位 −1.58mm / 有缝帧 3.6%
#     moving_jaw   中位 −1.88mm / **有缝帧 19.8%**
#     两爪同时贴合的帧仅 **78.5%**
# 即约 1/5 的帧活动爪根本没碰到方块，靠固定爪一侧顶着 + 摩擦带走 —— 用户目检所见。
#
# 于是 reward 与验收都改成几何量：爪面采样点到方块盒的有符号距离 ≤ 0 才算贴合。
# ★阈值必须给到**负值（真压入）**，不能设 0。设 0 时"gap≤0 即满分"，策略就停在边界上：
# 实测 v5（TOL=0）在 grasped 窗口里 gap 中位 **0.03mm**、只有 49.3% 的帧 ≤0
# —— 恰好卡在阈值上来回抖。要求压入 2mm 才给满分，策略才有理由真的收紧。
FIRM_GRASP_GAP_TOL = -0.002       # 满分阈值（米）：需压入 2mm
FIRM_GRASP_GAP_SCALE = 0.006      # 从满分阈值再放宽 6mm 处夹持分归零
# ★光有「压入多深」还不够，必须同时问「压在哪」——见 `_pinch_facts` 的实测：
# v7 有 84.8% 的搬运帧 gap<=0，却有 96.1% 的帧接触在棱上，真·两侧夹持仅 0.7%。
PINCH_EDGE_MARGIN = 0.70          # 接触点面内归一化位置上限（1.0=贴棱）；超过即不算夹住
PINCH_EDGE_SCALE = 0.25           # 从上限再放宽多少后中部分归零
# reward 用的连续量阈值：两爪接触点中点到质心的归一化距离
PINCH_CENTER_GOOD = 0.35          # 到此为止算「夹住质心」，给满分
PINCH_CENTER_SCALE = 0.55         # 再远这么多后中部分归零
# ★success 门用的阈值：比 reward 的满分线**宽**。
# 门设在满分线上会够不着：实测把门设成 `center_off≤0.35 ∧ gap≤-2mm` 后，策略在「已抬起」
# 帧上只有 **0.2%** 满足，正样本采不到 ⇒ success 从 0.31 塌到 0.008、128 集只剩 1 集。
# 而 `center_off≤0.50 ∧ gap≤0` 该策略能到 **39.0%**，且仍把擦棱的挡在外面
# （同判据下只改 reward 的上一版仅 14.1%）。⇒ reward 朝理想值优化，门取可达值。
GATE_CENTER_OFF = 0.50
GATE_GAP = 0.0
# 松手瞬间物体的速度上限（m/s）。真机是放到底、停稳再张爪，物体此时基本不动；
# 取 5cm/s 作为"已经躺住"的判据，横扫中途松手远超此值。
GENTLE_RELEASE_SPEED = 0.05
# 方块静置在箱底时的绝对高度（米）。实测 v7/v12 数据：落到箱底后 item z 稳定在 2.50cm
# （桌面上是 2.00cm）。料箱只在 xy 随机、z 不变，所以这个值是常量。
ITEM_REST_Z_IN_BIN = 0.025
# 「往箱底放」的引导强度与尺度。★这是 v13 缺的那一环：上一版只把「轻放」写进 success，
# **没给任何梯度** —— 策略永远撞不到那道墙，自然学不会。同当初「回家」那次的错误。
LOWER_REWARD_SCALE = 2.0
LOWER_REWARD_K = 20.0        # 高出箱底 5cm 时得 0.24 分、1cm 时 0.80 分、贴底 1.00 分
RELEASE_HIGH_PENALTY = 2.0   # 高处/带速度松手的惩罚
JAW_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")
JAW_SAMPLE_STRIDE = 24            # 爪网格顶点抽稀步长（每爪约几百点，够用且够快）


_JAW_VERTEX_CACHE = {}


def _jaw_sample_points(urdf_path, link_name):
    """某爪 link 的碰撞网格顶点（link 局部系，抽稀）。按 URDF 的 collision origin 变换。"""
    key = (str(urdf_path), link_name)
    if key in _JAW_VERTEX_CACHE:
        return _JAW_VERTEX_CACHE[key]

    text = Path(urdf_path).read_text()
    block = re.search(rf'<link name="{link_name}">(.*?)</link>', text, re.S)
    chunks = []
    for col in re.findall(r"<collision>(.*?)</collision>", block.group(1), re.S):
        origin = re.search(r'<origin xyz="([^"]+)"(?:\s+rpy="([^"]+)")?', col)
        mesh_ref = re.search(r'<mesh filename="([^"]+)"', col)
        if not mesh_ref:
            continue
        mesh_path = Path(urdf_path).parent / mesh_ref.group(1)
        if not mesh_path.exists():
            continue
        xyz = (np.array([float(x) for x in origin.group(1).split()])
               if origin else np.zeros(3))
        rpy = (np.array([float(x) for x in origin.group(2).split()])
               if (origin and origin.group(2)) else np.zeros(3))
        verts = np.asarray(trimesh.load(mesh_path, force="mesh").vertices,
                           float)[::JAW_SAMPLE_STRIDE]
        chunks.append((euler2mat(*rpy) @ verts.T).T + xyz)

    pts = np.concatenate(chunks)
    _JAW_VERTEX_CACHE[key] = pts
    return pts


class FirmGraspRewardMixin:
    """给 Place 的 reward 加一项「几何贴合」，并让搬运分以真夹住为前提。

    **为什么必须改 reward**：squint 的 Place 成功判据是
    `物体在料箱 x/y 内 ∧ 已松手 ∧ 物体静止 ∧ 机器人静止` —— **不含 `is_item_grasped`**。
    于是"托着走到箱口再丢进去"完全满足判据，策略没有任何理由把爪子收紧。

    **为什么判据必须是几何而不是力**：PhysX `contact_offset = 0.02` ⇒ 两体相距 2cm 以内
    就报接触力。实测两指各 16–18N 的同时，用两爪碰撞网格全部顶点量到方块盒的最小距离：
    `moving_jaw` **有缝帧占 19.8%**、两爪同时贴合仅 **78.5%** —— 约 1/5 的帧活动爪根本
    没碰到方块。所以力大小完全不能代表"夹住"，此前基于 `min(两指力)/5N` 的夹持项在
    2cm 内就能拿满分。

    三处改动（都在子类，不碰 vendored）：
    1. **几何贴合奖励**：两爪各取碰撞网格采样点，算到方块（有向盒）的有符号距离，
       取两爪中较差那一侧；≤0（压入）给满分，缝 `FIRM_GRASP_GAP_SCALE` 以上归零。
    2. **搬运分以夹住为前提**：vendored 的 `is_item_above_bin` 分支会**无条件**覆盖 reward
       （不要求 grasped），这正是"托着走"能拿高分的口子。未曾真贴合过则扣掉该增量。
    3. `evaluate()` 导出 `jaw_gap` / `is_firm_grasp`，供验收与常驻断言使用（几何量）。
    """

    def _jaw_points(self):
        """两爪采样点（link 局部系），首次调用时按 URDF 解析并缓存。"""
        if getattr(self, "_jaw_pts", None) is None:
            urdf = self.agent.urdf_path
            self._jaw_pts = {
                name: torch.as_tensor(_jaw_sample_points(urdf, name),
                                      dtype=torch.float32, device=self.device)
                for name in JAW_LINKS
            }
        return self._jaw_pts

    def _pinch_facts(self):
        """两爪最近点的 (较差 gap, 是否落在相对面, 较差面内偏移)。

        ★只看 `gap` 会瞎：它取所有采样点有符号距离的**最小值**，`argmin` 把「接触发生在
        哪儿」整个丢掉 —— 棱上蹭到 1mm 与侧面中部压入 1mm 同分。实测 v7 搬运帧（441 帧）：
        `gap<=0` 有 **84.8%**，但接触点**面内偏移中位 0.83、贴棱(>0.7)帧占 96.1%**，
        真·两侧夹持只有 **0.7%** —— 方块靠擦棱 + 摩擦被带走，正是用户目检的「像吸附」。

        所以再取两件事：接触点落在方块哪个面（法向轴 + 符号），以及它在该面内离棱多近
        （除法向轴外两个归一化坐标的较大者，1.0=贴棱）。
        """
        Ti = self.item.pose.to_transformation_matrix()
        Ri, pi = Ti[:, :3, :3], Ti[:, :3, 3]
        half = self.item_dimensions                      # (N,3) 半尺寸
        gaps, axes, signs, offs, norms = [], [], [], [], []
        for name, pts in self._jaw_points().items():
            T = self.agent.robot.links_map[name].pose.to_transformation_matrix()
            world = torch.einsum("nij,vj->nvi", T[:, :3, :3], pts) + T[:, :3, 3][:, None]
            local = torch.einsum("nij,nvi->nvj", Ri, world - pi[:, None])
            over = (local.abs() - half[:, None, :])
            outside = over.clamp(min=0).norm(dim=-1)     # 盒外欧氏距离
            inside = over.max(dim=-1).values             # 盒内为负
            signed = torch.where(outside > 0, outside, inside)
            best, idx = signed.min(dim=-1)
            pick = local[torch.arange(local.shape[0], device=local.device), idx]
            norm = pick / half                           # (N,3) 归一化到 ±1
            ax = norm.abs().argmax(dim=-1)
            sg = torch.sign(norm.gather(1, ax[:, None]).squeeze(1))
            on_axis = torch.nn.functional.one_hot(ax, norm.shape[-1]).bool()
            off = norm.abs().masked_fill(on_axis, -1.0).max(dim=-1).values
            gaps.append(best), axes.append(ax), signs.append(sg)
            offs.append(off), norms.append(norm)

        gap = torch.maximum(gaps[0], gaps[1])
        opposite = (axes[0] == axes[1]) & (signs[0] == -signs[1])
        # ★给 reward 用的连续量：两爪接触点的**中点**到方块质心的归一化距离。
        # 它一个数同时表达「相对」与「中部」——两爪都蹭在顶棱时中点偏到顶部（大），
        # 真从两侧夹住时中点≈质心（小）。比把 `opposite` 当硬门好：硬门早期恒假，
        # 夹持信号会稀疏到学不动。
        center_off = ((norms[0] + norms[1]) / 2).norm(dim=-1)
        return gap, opposite, torch.maximum(offs[0], offs[1]), center_off

    def _jaw_gap(self):
        """两爪到方块盒的最小有符号距离，取较差那一侧。<=0 = 两爪都贴合（**不含方位**）。"""
        return self._pinch_facts()[0]

    def compute_dense_reward(self, obs, action, info):
        reward = super().compute_dense_reward(obs, action, info)

        if "pinch_center_off" in info:
            gap, opposite = info["jaw_gap"], info["is_pinch_opposite"]
            edge_off, center_off = info["pinch_edge_off"], info["pinch_center_off"]
        else:
            gap, opposite, edge_off, center_off = self._pinch_facts()
        # 贴合分：gap<=tol 满分，线性衰减到 tol+scale 归零
        grip_score = (1.0 - (gap - FIRM_GRASP_GAP_TOL) / FIRM_GRASP_GAP_SCALE).clamp(0.0, 1.0)
        # ★再乘上「方位」两项，否则策略会去优化棱边接触（v7 就是这么学出来的）：
        #   中部分 —— 接触点离棱越远越高分；相对面 —— 两爪不在一对相对面上直接归零。
        centrality = (1.0 - (center_off - PINCH_CENTER_GOOD)
                      / PINCH_CENTER_SCALE).clamp(0.0, 1.0)
        grip_score = grip_score * centrality

        # ★这里必须用**可达**的门判据，不能用 `is_true_pinch` 那个又严又抖的量。
        # 下面 `above_bin_no_firm` 会按 `~_ever_firm` 每步扣 1.5 分，而"举到料箱上方"是
        # 完成任务的必经环节 —— 判据若几乎不可能为真（strict 版实测仅 0.2% 的帧满足），
        # 这一项就退化成**对放置动作本身的恒定惩罚**，策略学到的是别把物体举过去。
        # 该惩罚原本配的是宽判据（gap≤-2mm，可达 4–30%）；我加方位项时收严了 `firm_now`
        # 却没回头看它的下游用途，v11 成功率只有 0.062 极可能就是这么来的。
        firm_now = (gap <= GATE_GAP) & (center_off <= GATE_CENTER_OFF)
        if getattr(self, "_ever_firm", None) is None or self._ever_firm.shape[0] != firm_now.shape[0]:
            self._ever_firm = torch.zeros_like(firm_now)
        self._ever_firm |= firm_now

        # 贴合分只在**搬运途中**给。到了料箱上方就该松手，此时若还按住不放继续发分，
        # 举着每步稳拿 4+place+static+1≈7~8，而松手要先穿过物体下落、尚未静止的低分段
        # 才够得着 success 的 9 —— 在 gamma=0.96（视野约 25 步）下折算回来还不如举着。
        # 策略于是学出「夹稳后悬在箱口不放」：实测 331 帧不松手、jaw_gap 一路收紧到 -1.2mm。
        # 把箱上方那段的贴合分去掉，松手才是严格更优。
        grasped = info["is_item_grasped"]
        carrying = grasped & (~info["is_item_above_bin"])
        reward = reward + grip_score * carrying.float()
        above_bin_no_firm = info["is_item_above_bin"] & (~self._ever_firm)
        reward = reward - 1.5 * above_bin_no_firm.float()

        # ★放置引导（v14 新增）：夹着物体到了箱上方后，**越把物体往箱底放分越高**。
        # 真机就是这么做的——放到底、停稳、才张爪，全程不脱手（300 集实测松手前速度
        # 只有全程中位的 0.11，腕部画面里松手前 20 帧方块已躺在箱底）。
        # v13 只在 success 里加了「松手时物体近静止」，**没有梯度**，策略无从发现这条路径，
        # 实测 12/12 条成功轨迹全部在离箱底 3.8–9.2cm 的空中以 0.135–0.540 m/s 甩手，
        # 分布里连一条接近轻放的都没有 ⇒ 不是采样不足，是没给引导。
        above = info["is_item_above_bin"]
        holding = info["is_item_grasped"]
        height = (self.item.pose.p[:, 2] - ITEM_REST_Z_IN_BIN).clamp(min=0.0)
        lower_score = 1.0 - torch.tanh(LOWER_REWARD_K * height)
        reward = reward + LOWER_REWARD_SCALE * lower_score * (above & holding).float()

        # 高处或带速度松手要扣分，否则"举到箱上方直接丢"仍是捷径
        item_speed = torch.linalg.norm(self.item.linear_velocity, dim=-1)
        bad_release = above & (~info["robot_touching_item"]) & (
            (height > 0.01) | (item_speed > GENTLE_RELEASE_SPEED))
        reward = reward - RELEASE_HIGH_PENALTY * bad_release.float()
        return reward

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if getattr(self, "_ever_firm", None) is not None:
            self._ever_firm[env_idx] = False
        # 「本集曾真夹住」是 success 的必要条件，跨集必须清零，否则一集成功之后
        # 后面每集都白送这个条件（并行环境下尤其隐蔽：只有被 reset 的那几个 env 该清）
        if getattr(self, "_ever_center_grasp", None) is not None:
            self._ever_center_grasp[env_idx] = False

    def evaluate(self):
        info = super().evaluate()
        gap, opposite, edge_off, center_off = self._pinch_facts()
        info["jaw_gap"] = gap
        info["pinch_center_off"] = center_off
        info["is_pinch_opposite"] = opposite
        info["pinch_edge_off"] = edge_off
        # 旧口径：只问压没压到，不问压在哪（保留供对照，**不要拿它当"夹住了"**）
        info["is_firm_grasp"] = gap <= FIRM_GRASP_GAP_TOL
        # 报告口径：压入 ∧ 落在一对相对面 ∧ 都在面中部。
        # ⚠️ 只作报告，**不进 success**：面归属是 `argmax|坐标|` 定的，接触点靠近棱时
        # 3mm 的扰动就会翻面（可达性扫描实测 True/False 来回跳）⇒ 拿它当门会把噪声当信号。
        info["is_true_pinch"] = (info["is_firm_grasp"] & opposite
                                 & (edge_off <= PINCH_EDGE_MARGIN))
        # ★进 success 的口径用**连续量**：压入够深 ∧ 两爪接触点中点靠近方块质心。
        # 不含离散的面归属，故不抖；"中点靠近质心"本身已排除「两爪都蹭同一条棱」。
        # ★「放到底再松手」：真机 300 集实测，松手前 10 帧速度只有全程中位的 **0.11**、
        # 89% 的集明显减速；腕部画面里松手前 20 帧方块**已经躺在箱底**，前后几乎不动。
        # 即真机是「移到箱上方 → 一直放到底 → 停稳 → 张爪」，全程不脱手，**没有下落这一段**。
        # 我们的策略却是夹着横扫、边走边松手（实测水平位移 9.5cm > 落差 6.4cm，
        # "下落"段仍有 59% 的帧夹着，等效 g 仅 0.12–1.29 m/s²）。
        # 判据取**松手瞬间物体的速度**：放到底再张爪则≈0，横扫中途松手则很大。
        # 用速度而非高度，可避开料箱几何、也不必猜静置高度。
        item_speed = torch.linalg.norm(self.item.linear_velocity, dim=-1)
        info["item_speed"] = item_speed
        info["is_gentle_release"] = (~info["robot_touching_item"]) & (
            item_speed <= GENTLE_RELEASE_SPEED)

        center_grasp = (gap <= GATE_GAP) & (center_off <= GATE_CENTER_OFF)
        info["is_center_grasp"] = center_grasp
        if getattr(self, "_ever_center_grasp", None) is None \
                or self._ever_center_grasp.shape[0] != center_grasp.shape[0]:
            self._ever_center_grasp = torch.zeros_like(center_grasp)
        self._ever_center_grasp |= center_grasp

        # ★把「真夹住」变成成功的必要条件。
        # 不加这条，真夹住永远是可选项：squint 的 Place success =
        # 物体在箱内 ∧ 已松手 ∧ 静止，擦着棱把方块蹭进箱子照样满分，而夹持只是每步
        # 最多 +1 的附加奖励、抵不过省力路径。实测 v9（只改 reward 不改 success）
        # 搬运期 center_off 中位仍有 0.80、真夹持率 1.3% —— 与只改 reward 前几乎没变。
        # 同 `ReturnHomeMixin` 把「收回起始位姿」写进 success 的道理。
        info["placed"] = info["success"]          # 父类口径：放进去了（不问怎么夹的）
        # 成功 = 放进去了 ∧ 曾真夹住 ∧ 松手时物体已躺住（不是甩进去的）
        info["success"] = (info["success"] & self._ever_center_grasp
                           & info["is_gentle_release"])
        return info


class ReturnHomeMixin:
    """让一集包含「放完手再收回起始位姿」，策略自己学，不靠事后合成。

    真机采的 `task1` 每集都是从起始位姿出发、干完活回到起始位姿收尾（首末关节角最大只差
    11.8°）。而 squint 的 Place 判成功只要
    `物体在箱内 ∧ 没碰物体 ∧ 机器人静止 ∧ 没碰箱` —— **只要求"停住"，不要求"停在哪"**，
    停在料箱正上方照样算成功。实测出来的轨迹首末差中位 48.9°，机械臂就悬在箱口。

    三处改动，缺一不可：

    1. **成功判据加「回到起始位姿」**。不加这条，回家永远是可选项。
    2. **撤掉放手后那份「不动就给分」**。父类在 `is_item_above_bin` 段有
       `static_robot_reward = 1 - tanh(10|qvel|)`，停着不动稳拿满分；不撤掉它，
       启动回家反而先掉分。
    3. **回家要给稠密梯度，不能只靠终点大奖励**。gamma=0.96 视野约 25 步，而回家要走
       六七十步，终点那份奖励折算回放手时刻已经所剩无几（0.96^70≈0.06）。所以按
       「离起始位姿多近」逐帧给分，每挪一步都立刻变好，策略才跟得住。

    ★起始位姿取**每集 reset 后的实际 qpos**，不取 `start` keyframe 的名义值：两者对不上
    （keyframe 的 `wrist_roll` 是 −90°，实测稳定在 −67.2°，差 22.8°），拿名义值当目标会让
    「回到起始位姿」这句话在数据上不成立。
    """

    HOME_TOL = 0.15            # rad，关节空间 L2 距离（不含夹爪）；约合每关节几度
    HOME_REWARD_SCALE = 3.0    # 回家进度分的权重，要压过撤掉的那份 static 分
    SUCCESS_REWARD = 14.0      # 父类成功给 9；回家更长更难，终点分要相应抬高

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        qpos = self.agent.robot.get_qpos()
        if getattr(self, "_home_qpos", None) is None or self._home_qpos.shape != qpos.shape:
            self._home_qpos = qpos.clone()
        self._home_qpos[env_idx] = qpos[env_idx]

    def _latch_reachable_home(self):
        """把起始位姿改记成**第一步之后**的位姿。

        ★reset 那一刻的 qpos 不是物理可达的稳态：`start` keyframe 给的
        `wrist_roll = −90°`，第一个控制步就被拉到 −67.2° 并从此不动（实测一步跳 22.9°，
        零动作静置 60 步也回不去）。拿 reset 帧当回家目标，`home_dist` 有个 0.40 rad 的
        永久下界，远超 0.15 的容差 —— 机器人站着不动都算没回家，success 恒为 0。
        实测就是这么栽的：822k 步 `success_once` 严格 0.00，不是学不会，是够不着。
        """
        first_step = self.elapsed_steps == 1
        if bool(first_step.any()):
            qpos = self.agent.robot.get_qpos()
            self._home_qpos[first_step] = qpos[first_step]

    def _home_distance(self):
        """当前关节角到本集起始位姿的距离。夹爪不算——它开合由放手决定，不代表姿态回没回。"""
        qpos = self.agent.robot.get_qpos()
        return torch.linalg.norm(qpos[:, :-1] - self._home_qpos[:, :-1], dim=-1)

    def evaluate(self):
        info = super().evaluate()
        self._latch_reachable_home()
        home_dist = self._home_distance()
        info["home_dist"] = home_dist
        info["is_robot_home"] = home_dist <= self.HOME_TOL
        info["placed"] = info["success"]          # 父类口径：放进去了（不管停在哪）
        info["success"] = info["success"] & info["is_robot_home"]
        return info

    def compute_dense_reward(self, obs, action, info):
        reward = super().compute_dense_reward(obs, action, info)

        # 放开手之后才谈回家；还夹着就回去等于把物体带走
        released = info["is_item_above_bin"] & (~info["robot_touching_item"])

        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        home_score = 1 - torch.tanh(2.0 * info["home_dist"])

        # 撤掉「停着不动」那份，换成「离家多近」那份：停在原地不再有收益，往回挪立刻加分。
        # ★这里按 `released` 给、不按「还没到家」给：若一进容差圈就把这项撤掉，机械臂
        #   刚够到家、还没停稳（够不着 success）的那几帧会平白掉两分，成了回家路尽头的
        #   一个坑，策略宁可停在坑外。实测过一版就是这么掉的（9.33 -> 7.47）。
        adjust = self.HOME_REWARD_SCALE * home_score - static_robot_reward
        reward = reward + adjust * released.float()

        reward[info["success"]] = self.SUCCESS_REWARD
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_REWARD


class SlowRobotMixin:
    """把任务的机器人换成真机速度包线版，并提高物体摩擦（夹得更稳）+ 对齐真机 30fps。

    squint 的任务 `__init__` 只在 `robot_uids` 是 `so100`/`so101` 时才设 `base_z_rot` 与
    `rest_qpos`，换成别的 uid 会在 `_load_agent` 里报 `no attribute 'base_z_rot'`；
    这里像 `KitDualCameraMixin` 一样显式补上（`so101_slow` 与 `so101` 同构）。

    `control_freq` 设成 `SIM_FPS`（30）以对齐真机 task1；delta 上限在
    `so101_kit_slow.per_step_limits` 里按同一个 `SIM_FPS` 反算，故速度不随帧率漂。
    """

    SLOW_ROBOT_UID = "so101_slow"

    def __init__(self, *args, robot_uids=None, domain_randomization_config=None, **kwargs):
        self.base_z_rot = 0
        # 本 mixin 两种机器人都带：`so101_slow`（vanilla 几何）与子类覆盖的 `so101_kit_slow`
        # （KIT 几何）。wrist_roll 零位只在 KIT 那份上差 90°，故按 uid 分支，不能一刀切。
        self.rest_qpos = (kit_rest_qpos() if "kit" in self.SLOW_ROBOT_UID
                          else SO101.keyframes["start"].qpos.tolist())

        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        # 提高物体摩擦：3D 打印件配橡胶指垫，squint 默认 (0.1,0.5) 偏低导致"托着走"
        config.setdefault("item_friction_range", list(ITEM_FRICTION_RANGE))

        super().__init__(*args, robot_uids=self.SLOW_ROBOT_UID,
                         domain_randomization_config=config, **kwargs)

    @property
    def _default_sim_config(self):
        """把控制频率提到 30Hz 对齐真机 task1。

        ★`control_freq` 不是构造参数，走 `SimConfig`；且 **`sim_freq` 必须能整除
        `control_freq`**（`sapien_env` 里有 assert）。默认 `sim_freq=100` 除不尽 30，
        故同时把 `sim_freq` 提到 120（每个控制步 4 个物理子步，物理步长 1/120s）。
        """
        cfg = super()._default_sim_config
        cfg.sim_freq = 120
        cfg.control_freq = SIM_FPS
        return cfg


# ── KIT 几何 + 单 base_camera：训练任务 ────────────────────────────────
#
# ★为什么必须有这个：vanilla `so101.urdf` 与 KIT `kit_v1_so101.urdf` 的 `wrist_roll`
# 子系**零位恰差 90°**（关节轴 z 列相同 `[0,1,0]`，相对旋转 90.00°；其余关节 Δxyz/Δrpy
# 全为 0.000）。同一个 qpos 在两边把夹爪放到不同位置——`gripper_link` 差 6.0mm(z)、
# `moving_jaw` 差 38.1mm(x)/13.7mm(z)。于是"在 vanilla 训练 + 灌进 KIT 重渲"这条路
# 根本不自洽：策略按 vanilla 几何学会夹紧，画面与验收却按 KIT 几何算，永远对不上
# （实测 vanilla 侧 gap 中位 −1.53mm/贴合 94.6%，同批轨迹在 KIT 侧 +3.01mm/32.1%）。
# 真机几何不能改（用户 2026-07-31 明确），所以**训练必须直接用 KIT 几何**。
#
# 唯一障碍：KIT 双相机发 6 通道，而 squint 专家的 CNN 编码器是 3 通道。
# 解法=训练任务用 **KIT 机器人 + squint 默认的单 base_camera**（不套 KitDualCameraMixin），
# 双相机只在 replay 重渲时用。这样训练与验收共享同一套几何。


class KitGeomSingleCamMixin(SlowRobotMixin):
    """KIT 真机几何 + squint 默认单 base_camera（给专家训练/rollout 用）。

    只把机器人换成 KIT 版；相机沿用父环境默认，故观测是 3 通道，与专家编码器匹配。
    """

    SUPPORTED_ROBOTS = ["so100", "so101", "so101_slow", "so101_kit", "so101_kit_slow"]
    SLOW_ROBOT_UID = "so101_kit_slow"


# SPEED_SCALE 0.6->0.2 后每步位移只剩 1/3，同样动作要约 3 倍步数；上限随之从 400 提到 900。
# ── 纯 squint 配方 + 真值资产（数据集生产走这条）────────────────────────────
#
# 与上面那个的唯一差别：**不套 `FirmGraspRewardMixin`**，reward 与 success 完全用 squint 原逻辑。
#
# 为什么退回原配方：那套几何夹持奖励的前提是"件重 80g、指尖托不住"，而件其实是**海绵**
# （见 `ITEM_DENSITY` 处）。前提不成立，补丁就只剩副作用——五轮实测 v9→v14 成功率
# 0.31→0.008，每加一条约束塌一个量级，而 squint 原配方在其适用域内是能到 1.00 的。
#
# 起止的 home 段不在这里学（学回家曾让策略连取放都练不成，预算全耗在收臂上）：
# home→取放起点、松手→home 这两段是**两个已知关节构型之间的过渡**，用关节空间平滑插值
# 经控制器走出来即可，不需要逆解，也不需要策略去学。见 `hybrid/home_wrap.py`。
# 回家要多走六七十步，400 步的预算全被取放占掉会让策略压根没机会摸到成功
# ── 另外两种物料的 pick-and-place（数据集生产用）─────────────────────────────
#
# 与 cube_4 那对的差别只有物体：碰撞尺寸钉到各自 STEP 真值，视觉换成各自的 kit 网格。
# 20mm 方块与 40mm 圆柱都在夹爪行程内（削薄后的两指净空 −10°→2.4mm、20°→42.1mm）。
# 圆柱走 squint 的 `item_type="can"` 分支。
class KitSlowMixin(SlowRobotMixin, KitDualCameraMixin):
    """KIT 双相机 + 真机速度包线 + 高摩擦 + 30fps（重渲用）。

    继承 `SlowRobotMixin` 拿到摩擦/帧率/属性补齐三件事，只把机器人 uid 换成 KIT 版。
    """

    SUPPORTED_ROBOTS = KitDualCameraMixin.SUPPORTED_ROBOTS + ["so101_kit_slow"]
    SLOW_ROBOT_UID = "so101_kit_slow"


# `SO101KitGeomPlaceCube4RealPure-v1` 的双相机孪生：机器人 uid 同为 `so101_kit_slow`、
# 碰撞尺寸/密度/摩擦/撒点区逐字一致，**物理完全同构**，差别只在多一路 wrist 相机与换了渲染 mesh。
# 于是"在单相机环境里 rollout 录 action，再在这里按同一初始状态重放 action 出图"是确定性等价的，
# 这也正是 action-replay 验收门要检验的那件事。
# ── pick-and-place：抓起来 + 放进料箱 + 松手退开 ──────────────────────────
#
# 上面那批 Lift 只到"抓住并抬离"。Place 的成功判据强得多（squint 原逻辑）：
#   物体落在料箱 x/y 范围内 ∧ **夹爪已松开**(`~robot_touching_item`) ∧ 物体静止 ∧ 机器人静止
# 也就是"东西真进箱了、手真放开了、都停稳了"——这个用画面自证，比在腕部视角里辨接触可靠。
#
# 料箱默认尺寸（8×10×3cm）与 STEP `bin_2` 包围盒**逐字一致**，故不需要覆盖 bin 尺寸，
# 只把 item 碰撞盒钉到 STEP 真值。


# ══ 对外分发环境：一个场景一个环境 ═══════════════════════════════════════════
#
# 命名 `SO101<动词><物体><尺寸mm>-v<版本>`，名字里不带开发沿革（Kit/Geom/Real/Pure/Slow）。
# 每个场景**只有一个**环境，带完整双相机（top + wrist），录制与渲染都用它 ——
# 上面那些 `...Geom...` / `...Kit...` 成对出现的是历史遗留：两者用的是同一个机器人
# `so101_kit_slow`、同一套物理，差别只有相机集合，没有理由拆成两个。
# 合并的额外好处：录制端与渲染端物理**同一个环境**，不再存在"孪生环境是否逐字同构"的风险。
#
# 三个场景共用 `bin_2` 料箱（80×100×30mm，与 STEP 包围盒逐字一致）；
# 物体碰撞尺寸钉到各自 STEP 真值，不随机。
@register_env("SO101PickPlaceCube40-v1", max_episode_steps=400)
class SO101PickPlaceCube40(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin, Place):
    """把 40mm 方块抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cube_4"
    BIN_MESH = "bin_2"
    ITEM_SIZE_CONFIG = _cube_size_config("cube_4")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PickPlaceCube20-v1", max_episode_steps=400)
class SO101PickPlaceCube20(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin, Place):
    """把 20mm 方块抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cube_2"
    BIN_MESH = "bin_2"
    ITEM_SIZE_CONFIG = _cube_size_config("cube_2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PickPlaceCylinder40-v1", max_episode_steps=400)
class SO101PickPlaceCylinder40(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin,
                               Place):
    """把 40mm 圆柱抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cylinder_4"
    BIN_MESH = "bin_2"
    # ★圆柱 STEP 件的建模轴是 y，物理体的轴是 z（squint 用 euler2quat(0,π/2,0)
    #   把 `add_cylinder_collision` 的 x 轴转到 z）。缺这一行 ⇒ 视觉网格躺着渲染、
    #   物理体却立着，画面上就是「圆柱倒在桌上」。
    ITEM_MESH_ROTATION = _CYLINDER_Y_TO_Z
    ITEM_SIZE_CONFIG = _cylinder_size_config("cylinder_4")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)


# ══ 以下为历史遗留名，仅为兼容在跑的产线与既有 checkpoint 保留，新代码请用上面三个 ══

