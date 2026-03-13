from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

NavigationEventType = Literal[
    "goal_sent",
    "accepted",
    "feedback",
    "succeeded",
    "aborted",
    "canceled",
    "timeout",
    "connection_error",
]

TerminalNavigationState = Literal[
    "succeeded", "aborted", "canceled", "timeout", "connection_error"
]


@dataclass(frozen=True)
class NavigationPoint:
    x: float
    y: float
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def validate(self) -> None:
        fields = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }
        for name, value in fields.items():
            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                raise ValueError(f"'{name}' must be a finite number")


@dataclass(frozen=True)
class NavigationAdapterConfig:
    robot_host: str = "ole@192.168.10.10"
    goal_acceptance_timeout_s: float = 8.0
    goal_reached_timeout_s: float = 120.0
    retry_attempts: int = 0
    retry_on_states: tuple[TerminalNavigationState, ...] = (
        "connection_error",
        "timeout",
        "aborted",
    )
    cancel_on_timeout: bool = True


@dataclass(frozen=True)
class NavigationEvent:
    type: NavigationEventType
    attempt: int
    timestamp: float = field(default_factory=time.time)
    message: str | None = None
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class NavigationOutcome:
    terminal_state: TerminalNavigationState
    accepted: bool = False
    message: str | None = None


class NavigationTransport(Protocol):
    def send_goal(
        self,
        *,
        point: NavigationPoint,
        config: NavigationAdapterConfig,
        on_feedback: Callable[[dict[str, Any]], None],
    ) -> NavigationOutcome:
        ...

    def cancel_current_goal(self) -> None:
        ...


class Ros2CliNavigationTransport:
    """Transport via `ssh <host> ros2 action send_goal ...`.

    The adapter builds a structured NavigateToPose goal first and only serializes to
    a CLI payload at the transport boundary.
    """

    action_name = "/navigate_to_pose"
    action_type = "nav2_msgs/action/NavigateToPose"

    def __init__(self) -> None:
        self._last_process: subprocess.Popen[str] | None = None

    @staticmethod
    def build_goal_payload(point: NavigationPoint) -> dict[str, Any]:
        point.validate()
        return {
            "pose": {
                "header": {"frame_id": "map"},
                "pose": {
                    "position": {
                        "x": float(point.x),
                        "y": float(point.y),
                        "z": float(point.z),
                    },
                    "orientation": {
                        "x": float(point.qx),
                        "y": float(point.qy),
                        "z": float(point.qz),
                        "w": float(point.qw),
                    },
                },
            }
        }

    @classmethod
    def _build_command(cls, point: NavigationPoint, robot_host: str) -> list[str]:
        payload = json.dumps(cls.build_goal_payload(point), separators=(",", ":"))
        remote_cmd = [
            "ros2",
            "action",
            "send_goal",
            cls.action_name,
            cls.action_type,
            payload,
            "--feedback",
        ]
        return ["ssh", robot_host, *remote_cmd]

    def cancel_current_goal(self) -> None:
        process = self._last_process
        if process and process.poll() is None:
            process.terminate()

    def send_goal(
        self,
        *,
        point: NavigationPoint,
        config: NavigationAdapterConfig,
        on_feedback: Callable[[dict[str, Any]], None],
    ) -> NavigationOutcome:
        cmd = self._build_command(point, config.robot_host)
        try:
            self._last_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            return NavigationOutcome(
                terminal_state="connection_error", accepted=False, message=str(exc)
            )

        assert self._last_process.stdout is not None
        accepted = False
        start = time.monotonic()

        while True:
            line = self._last_process.stdout.readline()
            if line:
                line = line.strip()
                lower = line.lower()
                if "goal accepted" in lower:
                    accepted = True
                elif "feedback" in lower:
                    on_feedback({"raw": line})
                elif "succeeded" in lower:
                    return NavigationOutcome("succeeded", accepted=accepted)
                elif "aborted" in lower:
                    return NavigationOutcome("aborted", accepted=accepted, message=line)
                elif "canceled" in lower or "cancelled" in lower:
                    return NavigationOutcome("canceled", accepted=accepted, message=line)

            poll = self._last_process.poll()
            elapsed = time.monotonic() - start
            if not accepted and elapsed > config.goal_acceptance_timeout_s:
                return NavigationOutcome(
                    "timeout",
                    accepted=False,
                    message="Goal acceptance timeout exceeded",
                )
            if accepted and elapsed > config.goal_reached_timeout_s:
                return NavigationOutcome(
                    "timeout",
                    accepted=True,
                    message="Goal reached timeout exceeded",
                )
            if poll is not None:
                stderr = self._last_process.stderr.read().strip() if self._last_process.stderr else ""
                if poll == 0:
                    if accepted:
                        return NavigationOutcome("succeeded", accepted=True)
                    return NavigationOutcome("aborted", accepted=False, message=stderr)
                return NavigationOutcome(
                    "connection_error" if not accepted else "aborted",
                    accepted=accepted,
                    message=stderr or f"ros2 action exited with code {poll}",
                )


class NavigationAdapter:
    def __init__(
        self,
        *,
        transport: NavigationTransport | None = None,
        config: NavigationAdapterConfig | None = None,
    ) -> None:
        self.transport: NavigationTransport = transport or Ros2CliNavigationTransport()
        self.config = config or NavigationAdapterConfig()

    def navigate_to_point(
        self,
        point: NavigationPoint,
        *,
        timeout_s: float | None = None,
        on_event: Callable[[NavigationEvent], None] | None = None,
    ) -> TerminalNavigationState:
        run_config = (
            replace(self.config, goal_reached_timeout_s=timeout_s)
            if timeout_s is not None
            else self.config
        )
        event_handler = on_event or (lambda _event: None)
        attempts = run_config.retry_attempts + 1
        for attempt in range(1, attempts + 1):
            event_handler(NavigationEvent("goal_sent", attempt, data={"host": run_config.robot_host}))

            def _feedback_callback(payload: dict[str, Any]) -> None:
                event_handler(NavigationEvent("feedback", attempt, data=payload))

            outcome = self.transport.send_goal(
                point=point,
                config=run_config,
                on_feedback=_feedback_callback,
            )

            if outcome.accepted:
                event_handler(NavigationEvent("accepted", attempt, message=outcome.message))

            state = outcome.terminal_state
            event_handler(NavigationEvent(state, attempt, message=outcome.message))

            if state == "timeout" and run_config.cancel_on_timeout:
                self.transport.cancel_current_goal()
                event_handler(NavigationEvent("canceled", attempt, message="Canceled after timeout"))

            if state == "succeeded":
                return state
            if state not in run_config.retry_on_states or attempt >= attempts:
                return state

        return "aborted"
