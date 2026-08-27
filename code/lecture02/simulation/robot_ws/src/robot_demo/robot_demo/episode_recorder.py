"""episode_recorder：记录一次任务的完整过程到 JSONL 文件。

订阅 Topic：/target_joint、/joint_states、/action_command、/task_status
输出：JSONL 文件，每行一条记录（字段见下）。

字段：
    timestamp       unix 秒（float）
    joint_state     六关节当前位置（弧度）
    target_joint    六关节绝对目标（弧度）
    action_command  六关节位置增量（弧度）
    task_status     running / reached / timeout / stale_state
    success         是否成功（reached 为 True）
    error           错误原因（timeout / stale_state，否则 null）

以 /joint_states（20Hz 主时钟）驱动逐行写入；收到 reached/timeout 后写入
最终行并停止记录（success 由 task_status 推导）。
"""

import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from robot_demo.common import NUM_JOINTS


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("episode_recorder")
        self.declare_parameter("output_file", "/tmp/episode.jsonl")
        self.output_file = self.get_parameter("output_file").value

        self.target = None
        self.joint_state = None
        self.action = None
        self.status = "running"
        self.done = False
        self._final_written = False

        self.target_sub = self.create_subscription(
            Float64MultiArray, "/target_joint", self.on_target, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, 10
        )
        self.action_sub = self.create_subscription(
            Float64MultiArray, "/action_command", self.on_action, 10
        )
        self.status_sub = self.create_subscription(
            String, "/task_status", self.on_status, 10
        )

        self.file = open(self.output_file, "w")
        self.get_logger().info(
            f"episode_recorder started, writing to {self.output_file}"
        )

    def on_target(self, msg: Float64MultiArray):
        if len(msg.data) == NUM_JOINTS:
            self.target = list(msg.data)

    def on_joint_state(self, msg: JointState):
        if msg.position:
            self.joint_state = list(msg.position)
        self.write_record()

    def on_action(self, msg: Float64MultiArray):
        if len(msg.data) == NUM_JOINTS:
            self.action = list(msg.data)

    def on_status(self, msg: String):
        self.status = msg.data
        if msg.data in ("reached", "timeout"):
            self.done = True
        self.write_record()

    def write_record(self):
        if self.done and self._final_written:
            return

        success, error = self._compute_result()
        record = {
            "timestamp": time.time(),
            "joint_state": self.joint_state,
            "target_joint": self.target,
            "action_command": self.action,
            "task_status": self.status,
            "success": success,
            "error": error,
        }
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

        if self.done:
            self._final_written = True
            self.get_logger().info(
                f"episode finished: success={success}, error={error}"
            )

    def _compute_result(self):
        if self.status == "reached":
            return True, None
        if self.status == "timeout":
            return False, "timeout"
        if self.status == "stale_state":
            return False, "stale_state"
        return False, None

    def destroy_node(self):
        if not self.file.closed:
            self.file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
