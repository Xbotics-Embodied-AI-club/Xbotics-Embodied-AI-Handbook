"""task_status_node：根据反馈判断任务状态（running / reached / timeout / stale_state）。

订阅 Topic：/target_joint、/joint_states
发布 Topic：/task_status (std_msgs/String)

判断逻辑（本讲 README §6 约束 4/5，讲义 2.6.4）：
- stale_state：状态超过 200ms 未更新（新鲜度检查）。
- reached：最大关节误差 < 0.02 rad（成功来自反馈，绝不能以「命令已发送」代替）。
- timeout：超过最大执行时间仍未 reached。
- running：其余情况。

说明：新鲜度与超时都需要「周期性」检查才能正确触发（否则「状态停止更新」
本身就不会再触发回调），因此本节点用 10Hz 定时器驱动评估；/joint_states
回调只负责更新最新状态缓存。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from robot_demo.common import (
    NUM_JOINTS,
    REACH_TOLERANCE,
    STALE_STATE_NS,
    TASK_TIMEOUT,
)


class TaskStatusNode(Node):
    def __init__(self):
        super().__init__("task_status_node")
        self.status_pub = self.create_publisher(String, "/task_status", 10)

        self.target_sub = self.create_subscription(
            Float64MultiArray, "/target_joint", self.on_target, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, 10
        )

        self.target = None
        self.current = None
        self.last_state_time = None
        self.task_start_time = None
        self.finished = False

        # 10Hz 周期评估（新鲜度/到达/超时）
        self.create_timer(0.1, self.evaluate)
        self.get_logger().info("task_status_node started")

    def on_target(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warning(
                f"target dimension {len(msg.data)} != {NUM_JOINTS}, ignored"
            )
            return
        new_target = list(msg.data)
        # 目标未变化（例如 target_publisher 周期重复发布）则不重置任务，
        # 避免 reached 终态被反复清空。
        if self.target is not None and new_target == self.target:
            return
        self.target = new_target
        # 收到新目标：重置任务计时与终态
        self.task_start_time = self.get_clock().now()
        self.finished = False

    def on_joint_state(self, msg: JointState):
        if not msg.position:
            return
        self.current = list(msg.position)
        self.last_state_time = self.get_clock().now()

    def evaluate(self):
        if self.target is None or self.current is None:
            return
        if self.finished:
            return

        now = self.get_clock().now()

        # 1. 状态新鲜度检查（约束 4）
        if self.last_state_time is not None:
            state_age = now - self.last_state_time
            if state_age.nanoseconds > STALE_STATE_NS:
                self.publish_status("stale_state")
                return

        # 2. 到达判定（约束 5）
        reached = max(
            abs(target - current)
            for target, current in zip(self.target, self.current)
        ) < REACH_TOLERANCE
        if reached:
            self.finished = True
            self.publish_status("reached")
            return

        # 3. 超时判定（讲义 2.6.4）
        if self.task_start_time is not None:
            elapsed = now - self.task_start_time
            if elapsed.nanoseconds / 1e9 > TASK_TIMEOUT:
                self.finished = True
                self.publish_status("timeout")
                return

        self.publish_status("running")

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"task_status: {status}")


def main(args=None):
    rclpy.init(args=args)
    node = TaskStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
