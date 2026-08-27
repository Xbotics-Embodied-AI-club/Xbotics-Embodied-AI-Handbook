"""controller_node：校验并执行动作（更新 mock），不发布 /joint_states。

订阅 Topic：/action_command (std_msgs/Float64MultiArray)
发布 Topic：/mock_joint_increment (std_msgs/Float64MultiArray，内部 mock 机制)

职责（本讲 README §6 约束 3/6）：
- 校验动作维度必须等于六（约束 2）。
- 再做一次防御性单步限幅（clip 到 ±max_step），确保动作不可绕过控制器
  直接到达执行器（约束 3）。
- 把校验后的增量发布到内部 topic，由 robot_state_node 更新 mock。
  **本节点绝不发布 /joint_states**（约束 1）。

接真实硬件时：把「发布 /mock_joint_increment」替换为「向厂商控制器发送
经过验证的驱动命令」，校验与限幅逻辑保持不变。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robot_demo.common import MAX_STEP, NUM_JOINTS


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        self.action_sub = self.create_subscription(
            Float64MultiArray, "/action_command", self.on_action, 10
        )
        self.increment_pub = self.create_publisher(
            Float64MultiArray, "/mock_joint_increment", 10
        )
        self.get_logger().info("controller_node started")

    def on_action(self, msg: Float64MultiArray):
        increment = np.asarray(msg.data, dtype=float)
        if increment.shape[0] != NUM_JOINTS:
            self.get_logger().warning(
                f"action dimension {increment.shape[0]} != {NUM_JOINTS}, rejected"
            )
            return

        # 防御性单步限幅：即便上游（policy）出错，也不得单步越过 max_step
        increment = np.clip(increment, -MAX_STEP, MAX_STEP)

        out = Float64MultiArray()
        out.data = increment.tolist()
        self.increment_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
