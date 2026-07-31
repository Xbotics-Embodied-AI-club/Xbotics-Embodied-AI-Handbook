"""Minimal Embodied Agent skill loop for Lecture 17.

The script keeps the interfaces intentionally simple: an agent plan is a list of
named skills, each skill returns an observation, and the agent replans when an
observation reports failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    observation: str


Skill = Callable[[str], SkillResult]


def navigate_to(target: str) -> SkillResult:
    return SkillResult(True, f"arrived at {target}")


def detect_object(target: str) -> SkillResult:
    if target == "water bottle":
        return SkillResult(True, "water bottle detected on the table")
    return SkillResult(False, f"{target} not found")


def grasp(target: str) -> SkillResult:
    return SkillResult(False, f"first grasp attempt missed {target}")


def retry_grasp(target: str) -> SkillResult:
    return SkillResult(True, f"{target} secured after adjusted grasp")


def place(target: str) -> SkillResult:
    return SkillResult(True, f"{target} placed on the coffee table")


def check_success(target: str) -> SkillResult:
    return SkillResult(True, f"task success: {target} is on the coffee table")


SKILLS: dict[str, Skill] = {
    "navigate_to": navigate_to,
    "detect_object": detect_object,
    "grasp": grasp,
    "retry_grasp": retry_grasp,
    "place": place,
    "check_success": check_success,
}


def run_plan() -> None:
    task = "bring a water bottle from the kitchen to the living room table"
    plan: list[tuple[str, str]] = [
        ("navigate_to", "kitchen"),
        ("detect_object", "water bottle"),
        ("grasp", "water bottle"),
        ("place", "water bottle"),
        ("check_success", "water bottle"),
    ]

    print(f"task: {task}")
    step = 0

    while plan:
        step += 1
        skill_name, argument = plan.pop(0)
        print(f"\nthink[{step}]: call {skill_name}({argument!r})")

        skill = SKILLS[skill_name]
        result = skill(argument)
        status = "ok" if result.ok else "fail"
        print(f"act[{step}]: {skill_name}")
        print(f"observe[{step}]: {status} - {result.observation}")

        if result.ok:
            continue

        if skill_name == "grasp":
            print("update: add retry_grasp before continuing")
            plan.insert(0, ("retry_grasp", argument))
            continue

        raise RuntimeError(f"unrecoverable skill failure: {result.observation}")

    print("\nfinal: task completed with one recovered failure")


if __name__ == "__main__":
    run_plan()
