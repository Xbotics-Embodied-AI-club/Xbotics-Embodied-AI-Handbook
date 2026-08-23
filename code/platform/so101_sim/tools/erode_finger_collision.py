"""把夹爪两指的碰撞体**只在开合方向上变薄**，让两指能压进被夹物体一点点。

为什么需要：SO-101 的两个指尖沿插入方向**错开** 7.6mm（固定指尖在夹爪局部 z=−6.1mm，
动指尖在 18.5° 时 +1.5mm；要两尖齐平得张到约 35.5°，而那时指间距 59mm 已夹不住 40mm 方块）。
合拢时两个接触点不在同一高度，形成力偶，把**刚体**方块转着挤出爪外 —— 实测无论指令多深，
方块都被挤到只剩 10~15mm 咬合，搬运时还转 20°。真机夹的是海绵，压扁 4.7mm 后接触变成
大面积柔性贴合、力偶被吸收，所以能咬深又不转。

刚体方块与 STL 都不能改，那就改**碰撞近似**：把两指碰撞体沿开合方向各削薄 δ。于是
① 下降时两指与方块之间多出 δ 的净空，固定指不再骑到方块顶面上把手臂顶住；
② 合拢能比"勉强接触"再多合 2δ，方块被过盈固定，不再被挤出、也转不动。
视觉仍用原 STL；画面里就是两指压进方块一点点（渲染时方块画在前面遮住手指）。

**只沿开合方向削**，不做各向同性腐蚀 —— 后者会把手指削短、把薄片削没
（动指碰撞块沿 z 只有 6.9mm 厚，4mm 的等向腐蚀直接把它清空）。
开合方向取 `gripper_frame_link` 局部系的 x 轴，逐块换算到它自己的 link 系。

用法：`python erode_finger_collision.py <δ毫米>` → 覆盖 *_erode.obj
"""

import sys
from pathlib import Path

import numpy as np
import sapien
import trimesh
from transforms3d.quaternions import quat2mat

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "so101_sim/robots/kit_assets/assets"
URDF = ROOT / "so101_sim/robots/kit_assets/kit_v1_so101.urdf"
GRIP_REF_DEG = 16.0            # 换算开合方向时用的参考夹爪角（就是产线合拢的量级）
# 只削手指，不削舵机壳与腕部随动件（那些削了会让夹爪整体陷进桌面/料箱）
PARTS = {"gripper_link": ["Fixed_part_1.obj", "Fixed_part_2.obj"],
         "moving_jaw_so101_v1_link": [f"Moving_part_{k}.obj" for k in range(1, 11)]}


def open_axis_in_link_frames():
    """开合方向（gripper_frame 局部 x）在各 link 自身坐标系里的方向。"""
    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(str(URDF))
    pm = art.create_pinocchio_model()
    names = [ln.name for ln in art.get_links()]
    n = len(art.get_active_joints())
    q = np.zeros(n)
    q[:6] = np.radians([0.0, 0.0, 0.0, 90.0, 0.0, GRIP_REF_DEG])
    pm.compute_forward_kinematics(q)
    GR = quat2mat(np.asarray(pm.get_link_pose(names.index("gripper_frame_link")).q))
    out = {}
    for link in PARTS:
        LR = quat2mat(np.asarray(pm.get_link_pose(names.index(link)).q))
        out[link] = LR.T @ GR[:, 0]        # 开合方向在该 link 系里的单位向量
    return out


def thin_along(mesh, axis, delta):
    """沿 axis 方向把网格削薄 2*delta（绕质心等比压缩），其它方向不变。"""
    v = np.asarray(mesh.vertices, float)
    a = axis / np.linalg.norm(axis)
    t = v @ a
    c = 0.5 * (t.min() + t.max())
    half = 0.5 * (t.max() - t.min())
    keep = max(half - delta, 0.35 * half)          # 至少留 35% 厚度，别削成纸片
    v = v + np.outer((c + (t - c) * (keep / half)) - t, a)
    return trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False), half - keep


def main():
    delta = float(sys.argv[1]) / 1000.0
    axes = open_axis_in_link_frames()
    for link, files in PARTS.items():
        a = axes[link]
        for name in files:
            src = ASSETS / name
            if not src.exists():
                print(f"  跳过（没有）{name}")
                continue
            m = trimesh.load(src, force="mesh")
            e, cut = thin_along(m, a, delta)
            e.export(ASSETS / name.replace(".obj", "_erode.obj"))
            print(f"  {name:22s} 开合向厚度 −{cut*1000:.2f}mm  "
                  f"体积 {m.volume*1e6:6.2f} → {e.volume*1e6:6.2f}cm³")


if __name__ == "__main__":
    main()
