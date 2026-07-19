# PointCloud_Sandbox.py
# 功能：基于 Open3D GUI 的交互式点云滤波沙盒
# - 实时滑块控制：直通滤波 Z 阈值、体素降采样尺寸、统计滤波 K 值和标准差倍数
# - 滑块拖动即时更新 3D 渲染画面
# - 右侧面板实时显示原始/过滤后点数
# - 若无输入文件，自动生成带噪声的杯状测试点云（含桌面和飞点）
# - 适合教学演示和参数探索
#
# 依赖：numpy, open3d (需支持 gui 模块)

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os

class PointCloudSandbox:
    def __init__(self, file_path):
        # 1. 读取或自动生成测试数据
        self.prepare_data(file_path)
        
        # 2. 初始化 Open3D GUI 应用程序
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("具身智能感知前端：点云滤波沙盒", 1100, 750)
        
        # 3. 创建 3D 渲染画布 (Scene Widget)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background_color([0.12, 0.12, 0.14, 1.0]) # 暗色科技风背景
        
        # 4. 创建右侧控制面板容器 (Vertical Layout)
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        
        # --- UI 控件布局 ---
        self.panel.add_child(gui.Label("--- 1. 直通滤波 (Pass-Through) ---"))
        self.panel.add_child(gui.Label("保留 Z 轴小于该值的点 (米):"))
        self.slider_z = gui.Slider(gui.Slider.DOUBLE)
        self.slider_z.set_limits(0.2, 2.0)
        self.slider_z.double_value = 1.0
        self.slider_z.set_on_value_changed(self.on_param_change)
        self.panel.add_child(self.slider_z)
        
        self.panel.add_child(gui.Label("--- 2. 体素降采样 (Voxel Grid) ---"))
        self.panel.add_child(gui.Label("体素网格尺寸 Leaf Size (米):"))
        self.slider_voxel = gui.Slider(gui.Slider.DOUBLE)
        self.slider_voxel.set_limits(0.002, 0.04)
        self.slider_voxel.double_value = 0.01
        self.slider_voxel.set_on_value_changed(self.on_param_change)
        self.panel.add_child(self.slider_voxel)
        
        self.panel.add_child(gui.Label("--- 3. 统计滤波 (SOR 去噪) ---"))
        self.panel.add_child(gui.Label("检查的最近邻居数 (K):"))
        self.slider_k = gui.Slider(gui.Slider.INT)
        self.slider_k.set_limits(2, 50)
        self.slider_k.int_value = 20
        self.slider_k.set_on_value_changed(self.on_param_change)
        self.panel.add_child(self.slider_k)
        
        self.panel.add_child(gui.Label("标准差倍数 (Std Ratio):"))
        self.slider_std = gui.Slider(gui.Slider.DOUBLE)
        self.slider_std.set_limits(0.1, 4.0)
        self.slider_std.double_value = 2.0
        self.slider_std.set_on_value_changed(self.on_param_change)
        self.panel.add_child(self.slider_std)
        
        self.panel.add_child(gui.Label("---------------------------------"))
        self.info_label = gui.Label("点数统计计算中...")
        self.panel.add_child(self.info_label)
        
        # 5. 将画布和面板塞入窗口
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        
        # 6. 设置动态布局回调（拖动改变窗口大小时触发）
        self.window.set_on_layout(self.on_layout)
        
        # 7. 首次运行滤波流水线并聚焦视角
        self.update_pipeline()
        bounds = self.raw_pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

    def prepare_data(self, file_path):
        """确保有可读取的点云，若无则自动生成带噪声的杯状物体模型"""
        if os.path.exists(file_path):
            self.raw_pcd = o3d.io.read_point_cloud(file_path)
        else:
            print(f"未检测到本地 {file_path}，正在自动生成高逼真教学点云数据...")
            # 创建一个圆柱体作为杯身 + 基础平面作为桌面
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.08, height=0.2)
            cup_pts = cylinder.sample_points_uniformly(number_of_points=15000)
            # 整体平移，使其符合相机坐标系常识（Z为深度）
            cup_pts.translate([0, 0, 0.75])
            
            # 制造桌面和远端背景墙噪声
            table = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.6, depth=0.01)
            table_pts = table.sample_points_uniformly(number_of_points=10000)
            table_pts.translate([-0.3, -0.3, 0.85])
            
            # 制造离群空气飞点噪声
            noise = np.random.normal(0, 0.15, size=(800, 3)) + [0, 0, 0.8]
            noise_pcd = o3d.geometry.PointCloud()
            noise_pcd.points = o3d.utility.Vector3dVector(noise)
            
            self.raw_pcd = cup_pts + table_pts + noise_pcd
            self.raw_pcd.paint_uniform_color([0.2, 0.6, 0.9]) # 涂上科技蓝
            o3d.io.write_point_cloud(file_path, self.raw_pcd)

    def on_layout(self, layout_context):
        """划分左侧3D视窗与右侧240px的控制面板"""
        r = self.window.content_rect
        panel_width = 240
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    def on_param_change(self, value):
        """当任何一个滑动条被拖动时触发"""
        self.update_pipeline()

    def update_pipeline(self):
        """核心滤波流水线：读取滑块值 -> 运算 -> 重新渲染"""
        # 提取滑块当前的实时参数
        z_max = self.slider_z.double_value
        voxel_size = self.slider_voxel.double_value
        k_neighbors = int(self.slider_k.int_value)
        std_ratio = self.slider_std.double_value
        
        # 1. 直通滤波 (NumPy 快速切片)
        pts = np.asarray(self.raw_pcd.points)
        cols = np.asarray(self.raw_pcd.colors) if self.raw_pcd.has_colors() else None
        mask = pts[:, 2] <= z_max
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[mask])
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols[mask])
            
        # 2. 体素降采样
        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
        # 3. 统计滤波去噪
        if len(pcd.points) > k_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=k_neighbors, std_ratio=std_ratio)
            
        # 4. 刷新 3D 渲染画面
        self.scene_widget.scene.remove_geometry("filtered")
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.5  # 略微放大点尺寸方便观察
        self.scene_widget.scene.add_geometry("filtered", pcd, material)
        
        # 5. 动态更新文字看板
        self.info_label.text = f"原始点数: {len(self.raw_pcd.points):,}\n过滤后点数: {len(pcd.points):,}"

    def run(self):
        gui.Application.instance.run()

if __name__ == "__main__":
    # 实例化并运行沙盒
    sandbox = PointCloudSandbox("cup_raw.pcd")
    sandbox.run()