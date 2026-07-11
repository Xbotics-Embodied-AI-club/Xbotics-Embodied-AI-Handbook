from __future__ import annotations

from dataclasses import asdict

from robot_pick_place.config import get_task
from robot_pick_place.pose_generator import generate_targets


def main() -> None:
    for name in ("cube", "bottle"):
        task = get_task(name)
        targets = generate_targets(task)
        print(f"\n[{task.name}]")
        for target_name, pose in asdict(targets).items():
            values = ", ".join(f"{key}={value:.4f}" for key, value in pose.items())
            print(f"{target_name:>10}: {values}")


if __name__ == "__main__":
    main()
