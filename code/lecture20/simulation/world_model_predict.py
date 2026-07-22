"""Minimal one-step world model demo for Lecture 20.

The toy environment is a one-dimensional block. The "real" transition includes
friction, while the learned world model uses a slightly imperfect friction
estimate. The printed table makes prediction error visible without external
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    position: float
    velocity: float


def real_step(state: State, action: float) -> State:
    velocity = 0.70 * state.velocity + action
    position = state.position + velocity
    return State(position=position, velocity=velocity)


def model_predict(state: State, action: float) -> State:
    predicted_velocity = 0.82 * state.velocity + action
    predicted_position = state.position + predicted_velocity
    return State(position=predicted_position, velocity=predicted_velocity)


def rollout() -> None:
    state = State(position=0.0, velocity=0.0)
    actions = [0.35, 0.20, -0.10, 0.05, 0.00]

    print("step | action | actual_pos | pred_pos | abs_error")
    print("-----|--------|------------|----------|----------")

    total_error = 0.0
    for step, action in enumerate(actions, start=1):
        prediction = model_predict(state, action)
        actual = real_step(state, action)
        error = abs(actual.position - prediction.position)
        total_error += error

        print(
            f"{step:>4} | {action:>6.2f} | {actual.position:>10.3f} | "
            f"{prediction.position:>8.3f} | {error:>8.3f}"
        )
        state = actual

    print(f"\nmean one-step position error: {total_error / len(actions):.3f}")


if __name__ == "__main__":
    rollout()
