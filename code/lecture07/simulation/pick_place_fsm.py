"""No-hardware entry for Lecture 07 pick-place FSM (MockBackend)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python simulation/pick_place_fsm.py` from code/lecture07/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_pick_place.run_demo import build_parser, main as run_main  # noqa: E402


def main() -> int:
    # Reuse the package CLI so docs and scripts stay in sync.
    return run_main()


if __name__ == "__main__":
    # Ensure argparse still sees --task when launched as a script.
    raise SystemExit(main())
