from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Any

from .models import Observation, State


class TaskLogger:
    def __init__(self, output_dir: str | Path, task_name: str) -> None:
        stamp = time_ns_text()
        self.task_dir = Path(output_dir) / f"{stamp}_{task_name}"
        self.task_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.task_dir / "events.jsonl"

    def log(
        self,
        state: State,
        event: str,
        observation: Observation | None = None,
        **extra: Any,
    ) -> None:
        record: dict[str, Any] = {
            "time": time(),
            "state": state.name,
            "event": event,
            **extra,
        }
        if observation is not None:
            record["observation"] = asdict(observation)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def time_ns_text() -> str:
    return str(int(time() * 1_000_000))
