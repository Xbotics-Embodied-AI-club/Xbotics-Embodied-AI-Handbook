"""minimal_loop.launch.py：一键启动 robot_demo 全部 6 个节点。

用法：
    ros2 launch robot_demo minimal_loop.launch.py
    ros2 launch robot_demo minimal_loop.launch.py output_file:=~/robot_ws/episode.jsonl

对比讲义 2.6 的片段，此处补上了 output 与 episode_recorder 的
记录路径参数；节点集合与职责完全一致。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    output_file_arg = DeclareLaunchArgument(
        "output_file",
        default_value="/tmp/episode.jsonl",
        description="Path to the episode JSONL output file",
    )

    target_publisher = Node(
        package="robot_demo",
        executable="target_publisher",
        output="screen",
    )
    robot_state_node = Node(
        package="robot_demo",
        executable="robot_state_node",
        output="screen",
    )
    policy_node = Node(
        package="robot_demo",
        executable="policy_node",
        output="screen",
    )
    controller_node = Node(
        package="robot_demo",
        executable="controller_node",
        output="screen",
    )
    task_status_node = Node(
        package="robot_demo",
        executable="task_status_node",
        output="screen",
    )
    episode_recorder = Node(
        package="robot_demo",
        executable="episode_recorder",
        output="screen",
        parameters=[{"output_file": LaunchConfiguration("output_file")}],
    )

    return LaunchDescription([
        output_file_arg,
        target_publisher,
        robot_state_node,
        policy_node,
        controller_node,
        task_status_node,
        episode_recorder,
    ])
