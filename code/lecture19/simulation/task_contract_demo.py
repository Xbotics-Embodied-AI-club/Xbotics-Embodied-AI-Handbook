"""一个不依赖第三方库的最小 Embodied Agent 架构教程。

程序使用预先编写的决策，让初学者先关注任务合同、证据、安全门和执行预算。
以后可以把规则决策替换成 LLM，而不改变这些执行边界。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SuccessCriterion:
    subject: str
    predicate: str
    object: str
    max_age_sec: float = 60.0


@dataclass(frozen=True)
class EvidenceFact:
    subject: str
    predicate: str
    object: str
    observed_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TaskContract:
    objective: str
    success_criteria: tuple[SuccessCriterion, ...]

    def __post_init__(self) -> None:
        if not self.success_criteria:
            raise ValueError("a task must define at least one success criterion")


@dataclass(frozen=True)
class SkillContract:
    name: str
    required_arguments: tuple[str, ...]
    preconditions: tuple[str, ...]
    success_criteria: tuple[str, ...]
    failure_modes: tuple[str, ...]
    safety_level: str = "normal"


@dataclass(frozen=True)
class SkillResult:
    success: bool
    summary: str
    evidence: tuple[EvidenceFact, ...] = ()
    failure_mode: str | None = None


@dataclass(frozen=True)
class RobotStatus:
    online: bool = True
    battery_percentage: float = 100.0
    emergency_stop: bool = False
    collision_detected: bool = False


SKILLS = {
    "inspect_scene": SkillContract(
        name="inspect_scene",
        required_arguments=("question",),
        preconditions=("camera_available",),
        success_criteria=("a fresh observation is returned",),
        failure_modes=("camera_unavailable", "no_target_found"),
    ),
    "pick": SkillContract(
        name="pick",
        required_arguments=("object_id",),
        preconditions=("object_visible", "object_reachable", "gripper_empty"),
        success_criteria=("object is held by robot",),
        failure_modes=("object_not_found", "grasp_missed", "force_limit"),
        safety_level="motion",
    ),
    "place": SkillContract(
        name="place",
        required_arguments=("object_id", "target_id"),
        preconditions=("object_in_gripper", "target_reachable"),
        success_criteria=("object is inside target region",),
        failure_modes=("target_unreachable", "object_fell"),
        safety_level="motion",
    ),
}


class SkillGateway:
    """在物理执行前验证 SkillIntent。"""

    @staticmethod
    def validate(
        skill_name: str, arguments: dict[str, str], status: RobotStatus
    ) -> tuple[bool, str]:
        contract = SKILLS.get(skill_name)
        if contract is None:
            return False, "unknown_skill"
        missing = [key for key in contract.required_arguments if key not in arguments]
        if missing:
            return False, f"missing_arguments:{','.join(missing)}"
        if not status.online:
            return False, "robot_offline"
        if status.emergency_stop:
            return False, "emergency_stop_active"
        if status.collision_detected:
            return False, "collision_detected"
        if contract.safety_level == "motion" and status.battery_percentage < 20.0:
            return False, "battery_too_low_for_motion"
        return True, "accepted"


class TaskEvaluator:
    """只有每项成功条件都有新鲜证据时，才判定任务完成。"""

    @staticmethod
    def evaluate(
        contract: TaskContract, evidence: tuple[EvidenceFact, ...]
    ) -> tuple[str, tuple[SuccessCriterion, ...]]:
        now = time.time()
        missing = tuple(
            criterion
            for criterion in contract.success_criteria
            if not any(
                fact.subject == criterion.subject
                and fact.predicate == criterion.predicate
                and fact.object == criterion.object
                and now - fact.observed_at <= criterion.max_age_sec
                for fact in evidence
            )
        )
        return ("satisfied", ()) if not missing else ("inconclusive", missing)


REASON_ZH = {
    "accepted": "通过",
    "unknown_skill": "Skill 不存在",
    "robot_offline": "机器人离线",
    "emergency_stop_active": "急停已触发",
    "collision_detected": "检测到碰撞",
    "battery_too_low_for_motion": "电量过低，禁止运动",
}


def reason_zh(reason: str) -> str:
    if reason.startswith("missing_arguments:"):
        fields = reason.split(":", 1)[1]
        return f"缺少必需参数：{fields}"
    return REASON_ZH.get(reason, reason)


def bool_zh(value: bool) -> str:
    return "是" if value else "否"


def print_skill_intent(skill: str, arguments: dict[str, str]) -> None:
    print(f"Agent：请求 request_skill({skill!r}, {arguments})")


def scenario_success() -> None:
    print("\n=== 场景一：Skill 成功，并且 Evidence 完整 ===")
    task = TaskContract(
        "把水瓶放到客厅茶几上",
        (SuccessCriterion("water_bottle", "at", "coffee_table"),),
    )
    status = RobotStatus()
    arguments = {"object_id": "water_bottle", "target_id": "coffee_table"}
    print(f"TaskContract：{task.objective}")
    print_skill_intent("place", arguments)
    allowed, reason = SkillGateway.validate("place", arguments, status)
    print(f"SkillGateway：允许={bool_zh(allowed)}，原因={reason_zh(reason)}（{reason}）")
    assert allowed
    result = SkillResult(
        True,
        "放置动作已经结束",
        (EvidenceFact("water_bottle", "at", "coffee_table"),),
    )
    print(f"SkillResult：成功={bool_zh(result.success)}，摘要={result.summary!r}")
    evaluation, missing = TaskEvaluator.evaluate(task, result.evidence)
    print(f"Evaluator：状态=已满足（{evaluation}），缺失条件={len(missing)}")
    assert evaluation == "satisfied"


def scenario_missing_evidence() -> None:
    print("\n=== 场景二：Skill 成功，但缺少 Evidence ===")
    task = TaskContract(
        "把水瓶放到客厅茶几上",
        (SuccessCriterion("water_bottle", "at", "coffee_table"),),
    )
    arguments = {"object_id": "water_bottle", "target_id": "coffee_table"}
    print_skill_intent("place", arguments)
    allowed, reason = SkillGateway.validate("place", arguments, RobotStatus())
    print(f"SkillGateway：允许={bool_zh(allowed)}，原因={reason_zh(reason)}（{reason}）")
    assert allowed
    result = SkillResult(True, "place 函数正常返回")
    print(f"SkillResult：成功={bool_zh(result.success)}，Evidence 数量={len(result.evidence)}")
    evaluation, missing = TaskEvaluator.evaluate(task, result.evidence)
    print(f"Evaluator：状态=证据不足（{evaluation}），缺失条件={len(missing)}")
    assert evaluation == "inconclusive"
    print("更新计划：request_observation('水瓶真的在茶几上吗？')")


def scenario_low_battery() -> None:
    print("\n=== 场景三：低电量阻止机器人运动 ===")
    status = RobotStatus(battery_percentage=5.0)
    arguments = {"object_id": "water_bottle"}
    print_skill_intent("pick", arguments)
    allowed, reason = SkillGateway.validate("pick", arguments, status)
    print(f"SkillGateway：允许={bool_zh(allowed)}，原因={reason_zh(reason)}（{reason}）")
    assert not allowed
    print("Robot Runtime：没有收到 RobotAction，机器人不会运动")


def scenario_budget_exhausted() -> None:
    print("\n=== 场景四：重复搜索达到执行预算 ===")
    task = TaskContract(
        "寻找一个并不存在的目标",
        (SuccessCriterion("unicorn", "visible_in", "room"),),
    )
    evidence: tuple[EvidenceFact, ...] = ()
    max_deliberations = 3
    for turn in range(1, max_deliberations + 1):
        print(f"第 {turn} 轮：request_observation('寻找独角兽')")
        print("Observation：没有找到目标（target_not_found）")
        evaluation, _ = TaskEvaluator.evaluate(task, evidence)
        print(f"Evaluator：证据不足（{evaluation}）")
    print(f"执行预算：经过 {max_deliberations} 轮推理后已经耗尽")
    print("任务状态：失败；停止无限搜索并向人类求助")


SCENARIOS = {
    "success": scenario_success,
    "missing-evidence": scenario_missing_evidence,
    "low-battery": scenario_low_battery,
    "budget": scenario_budget_exhausted,
}


def main() -> None:
    requested = sys.argv[1:] or list(SCENARIOS)
    unknown = [name for name in requested if name not in SCENARIOS]
    if unknown:
        choices = ", ".join(SCENARIOS)
        raise SystemExit(f"未知场景 {unknown[0]!r}；可选值：{choices}")
    for name in requested:
        SCENARIOS[name]()


if __name__ == "__main__":
    main()
