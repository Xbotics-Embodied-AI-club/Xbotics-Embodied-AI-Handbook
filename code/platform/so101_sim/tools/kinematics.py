"""URDF 正运动学的最小公共件：给量测类工具用，不参与仿真、不进训练循环。

只提供两样东西：
- `ArmKinematics`：挂一份 CPU-only 的 URDF 副本（`sapien` 的 pinocchio 绑定），
  能对给定关节角做正运动学，取任意 link 的位姿。GPU 后端下 `create_pinocchio_model()`
  直接抛 `NotImplementedError`，所以量测代码要单独起一个 CPU 场景来算 FK。
- `BaseFrame`：机器人根连杆在世界系下的位姿，把 CPU 副本算出的**基座系**坐标转到
  **世界系**——机器人不一定摆在世界原点，漏转会导致数字整体偏出一个固定量。

这里只保留量测用得到的部分。伺服控制那套（雅可比、逆解、末端/指尖局部位姿）不在此列：
量测只需要"给关节角、要 link 位姿"这一个方向，把控制相关的一并搬进来会让工具变重。
"""

import numpy as np
import sapien
from transforms3d.quaternions import quat2mat


class ArmKinematics:
    """挂在仿真旁边的一份 CPU-only URDF 副本，只做正运动学。"""

    def __init__(self, urdf_path):
        self._scene = sapien.Scene()
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._art = loader.load(str(urdf_path))
        self._pm = self._art.create_pinocchio_model()
        self.n = len(self._art.get_active_joints())


class BaseFrame:
    """机器人根连杆在世界系下的位姿，用来把基座系坐标转到世界系。"""

    def __init__(self, pose_p, pose_q):
        self.p = np.asarray(pose_p, float)
        self.R = quat2mat(np.asarray(pose_q, float))

    def to_world(self, p_local):
        return self.R @ np.asarray(p_local, float) + self.p
