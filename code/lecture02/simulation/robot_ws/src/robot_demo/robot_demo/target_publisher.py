"""target_publisher：发布六关节绝对目标位置。

发布 Topic：/target_joint (std_msgs/Float64MultiArray)
语义：六关节**绝对目标**，弧度，顺序与 common.JOINT_NAMES 一致。

以固定周期重复发布，保证晚启动的订阅者（policy_node / task_status_node /
episode_recorder）也能收到目标。
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robot_demo.common import DEFAULT_TARGET, TARGET_PERIOD


class TargetPublisher(Node):
    def __init__(self):
        super().__init__("target_publisher")
        self.target_pub = self.create_publisher(
            Float64MultiArray, "/target_joint", 10
        )
        self.create_timer(TARGET_PERIOD, self.timer_callback)
        self.get_logger().info("target_publisher started")

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = list(DEFAULT_TARGET)
        self.target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
