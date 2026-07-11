from __future__ import annotations

import csv
import json
from pathlib import Path


def summarize_task(task_dir: Path) -> dict[str, object]:
    events_path = task_dir / "events.jsonl"
    records = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    success = any(record.get("event") == "success" for record in records)
    failures = [record for record in records if record.get("event") == "failure"]
    states = [record["state"] for record in records if record.get("event") == "enter"]
    return {
        "task_dir": task_dir.name,
        "success": success,
        "failure_count": len(failures),
        "last_failure": failures[-1].get("code", "") if failures else "",
        "state_count": len(states),
    }


def main() -> None:
    root = Path("runs")
    summaries = [summarize_task(path) for path in sorted(root.iterdir()) if path.is_dir()]
    output = root / "summary.csv"
    if not summaries:
        raise SystemExit("no task directories found under runs/")
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(output)


if __name__ == "__main__":
    main()
