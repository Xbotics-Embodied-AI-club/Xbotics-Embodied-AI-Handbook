import os
from glob import glob

from setuptools import find_packages, setup

package_name = "robot_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description=(
        "Lecture 02 demo: a minimal six-joint robot closed loop over ROS2 "
        "(target -> policy -> controller -> state -> status)."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "target_publisher = robot_demo.target_publisher:main",
            "robot_state_node = robot_demo.robot_state_node:main",
            "policy_node = robot_demo.policy_node:main",
            "controller_node = robot_demo.controller_node:main",
            "task_status_node = robot_demo.task_status_node:main",
            "episode_recorder = robot_demo.episode_recorder:main",
        ],
    },
)
