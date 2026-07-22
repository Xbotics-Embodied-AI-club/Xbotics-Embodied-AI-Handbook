from __future__ import annotations

import argparse

from .backends.mock import MockBackend
from .config import get_task
from .state_machine import PickPlaceStateMachine
from .task_logger import TaskLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cube", "bottle"], default="cube")
    parser.add_argument("--output", default="runs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task = get_task(args.task)
    backend = MockBackend()
    logger = TaskLogger(args.output, task.name)

    backend.connect()
    try:
        machine = PickPlaceStateMachine(backend, task, logger)
        success = machine.run()
    finally:
        backend.disconnect()

    print(f"task={task.name}, success={success}")
    print(f"log_dir={logger.task_dir}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
