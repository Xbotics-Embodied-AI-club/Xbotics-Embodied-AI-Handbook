"""robot_state_node：维护 mock 关节位置并发布 /joint_states（唯一发布者）。

发布 Topic：/joint_states (sensor_msgs/JointState)
订阅 Topic：/mock_joint_increment (std_msgs/Float64MultiArray，内部 mock 机制)

职责：
- 维护内存中的 mock 关节位置（初始全 0），按固定频率（20Hz）发布 JointState，
  含 position / header.stamp / 固定顺序的 name。
- 订阅 /mock_joint_increment（由 controller_node 发布），把增量加到 mock 位置
  并做关节限位 clip。

约束（本讲 README §6 约束 1）：/joint_states 只有本节点发布，controller_node 不得
伪造第二路状态。接真实硬件时，仅需把 mock 读取替换为真实关节反馈读取，
Topic 保持不变。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robot_demo.common import JOINT_LIMITS, JOINT_NAMES, NUM_JOINTS, STATE_RATE


class RobotStateNode(Node):
    def __init__(self):
        super().__init__("robot_state_node")
        self.joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self.increment_sub = self.create_subscription(
            Float64MultiArray, "/mock_joint_increment", self.on_increment, 10
        )

        self.mock_position = np.zeros(NUM_JOINTS)
        self.create_timer(1.0 / STATE_RATE, self.timer_callback)
        self.get_logger().info("robot_state_node started")

    def on_increment(self, msg: Float64MultiArray):
        increment = np.asarray(msg.data, dtype=float)
        if increment.shape[0] != NUM_JOINTS:
            self.get_logger().warning(
                f"increment dimension {increment.shape[0]} != {NUM_JOINTS}, ignored"
            )
            return
        lower, upper = JOINT_LIMITS
        self.mock_position = np.clip(
            self.mock_position + increment, lower, upper
        )

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = self.mock_position.tolist()
        # mock 无速度模型，速度恒为 0
        msg.velocity = [0.0] * NUM_JOINTS
        self.joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
