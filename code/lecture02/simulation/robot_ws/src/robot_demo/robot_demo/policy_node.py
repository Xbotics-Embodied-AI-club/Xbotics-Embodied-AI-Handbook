"""policy_node：由目标误差生成六关节位置增量。

订阅 Topic：/target_joint、/joint_states
发布 Topic：/action_command (std_msgs/Float64MultiArray)

语义（本讲最容易错的对齐点）：/target_joint 是绝对目标，/action_command 是
位置增量。两者单位相同（弧度）、语义不同——绝不能把增量当绝对目标下发。

策略：action = clip(gain * (target - current), -max_step, max_step)
（对应本讲 README §6 约束 3 的动作单步限幅。）
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robot_demo.common import GAIN, MAX_STEP, NUM_JOINTS


def build_action(current, target, gain=GAIN, max_step=MAX_STEP):
    """根据当前与目标关节位置计算位置增量（带单步限幅）。

    current / target：六关节位置序列（绝对位置，弧度）。
    返回：np.ndarray，形状与输入一致，每个分量被 clip 到 [-max_step, max_step]。
    """
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")

    error = target - current
    return np.clip(gain * error, -max_step, max_step)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.target = None

        self.target_sub = self.create_subscription(
            Float64MultiArray, "/target_joint", self.on_target, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, 10
        )
        self.action_pub = self.create_publisher(
            Float64MultiArray, "/action_command", 10
        )
        self.get_logger().info("policy_node started")

    def on_target(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warning(
                f"target dimension {len(msg.data)} != {NUM_JOINTS}, ignored"
            )
            return
        self.target = list(msg.data)

    def on_joint_state(self, msg: JointState):
        if not msg.position or self.target is None:
            return
        try:
            action = build_action(msg.position, self.target)
        except ValueError as exc:
            self.get_logger().warning(f"cannot build action: {exc}")
            return

        command = Float64MultiArray()
        command.data = action.tolist()
        self.action_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
