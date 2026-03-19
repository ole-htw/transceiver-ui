from __future__ import annotations

import json
import math
import re
import shlex
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

NavigationEventType = Literal[
    "goal_sent",
    "accepted",
    "feedback",
    "position",
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
    ros2_namespace: str = ""
    ros2_action_name: str = "/navigate_to_pose"
    remote_ros_env_cmd: str = ""
    remote_ros_setup: str = ""
    fastdds_profiles_file: str = ""
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

    action_type = "nav2_msgs/action/NavigateToPose"

    def __init__(self) -> None:
        self._last_process: subprocess.Popen[str] | None = None
        self._last_config: NavigationAdapterConfig | None = None
        self._last_goal_id: str | None = None
        self._goal_id_pattern = re.compile(
            r"(?:goal(?:[_\s-]?id)?|id)\s*[:=]\s*([0-9a-fA-F-]{36})"
        )
        self._position_line_pattern = re.compile(
            r"position\s*[:=]?\s*(?:\{)?[^{}]*\bx\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r".*?\by\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        self._yaw_pattern = re.compile(
            r"\byaw\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        self._frame_id_pattern = re.compile(
            r"\bframe_id\s*[:=]\s*['\"]?([A-Za-z0-9_./-]+)['\"]?",
            re.IGNORECASE,
        )

    @staticmethod
    def _tail_text(text: str, *, max_lines: int = 20) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        return "\\n".join(lines[-max_lines:])

    def _extract_position_payload(self, line: str) -> dict[str, Any] | None:
        match = self._position_line_pattern.search(line)
        if not match:
            return None
        try:
            x = float(match.group(1))
            y = float(match.group(2))
        except (TypeError, ValueError):
            return None
        yaw_match = self._yaw_pattern.search(line)
        yaw = float(yaw_match.group(1)) if yaw_match else None
        frame_match = self._frame_id_pattern.search(line)
        frame_id = frame_match.group(1) if frame_match else "map"
        return {"x": x, "y": y, "yaw": yaw, "frame_id": frame_id}

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

    @staticmethod
    def _resolve_action_name(*, namespace: str, action_name: str) -> str:
        ns = namespace.strip("/")
        action = action_name.strip()
        if not action.startswith("/"):
            action = f"/{action}"
        if not ns:
            return action
        return f"/{ns}{action}"

    @staticmethod
    def _build_env_source_label(config: NavigationAdapterConfig) -> str:
        if config.remote_ros_env_cmd.strip():
            return "TRANSCEIVER_REMOTE_ROS_ENV_CMD"
        if config.remote_ros_setup.strip():
            return "TRANSCEIVER_REMOTE_ROS_SETUP"
        return "none"

    @classmethod
    def _build_command(cls, point: NavigationPoint, config: NavigationAdapterConfig) -> list[str]:
        resolved_namespace = config.ros2_namespace.strip("/")

        payload = json.dumps(cls.build_goal_payload(point), separators=(",", ":"))
        resolved_action = cls._resolve_action_name(
            namespace=config.ros2_namespace,
            action_name=config.ros2_action_name,
        )
        ros2_cmd = " ".join(
            [
                "ros2",
                "action",
                "send_goal",
                shlex.quote(resolved_action),
                shlex.quote(cls.action_type),
                shlex.quote(payload),
                "--feedback",
            ]
        )
        remote_ros_env_cmd = config.remote_ros_env_cmd.strip()
        remote_setup = config.remote_ros_setup.strip()
        fastdds_profiles_file = config.fastdds_profiles_file.strip()

        env_exports: list[str] = []
        if fastdds_profiles_file:
            quoted_profile = shlex.quote(fastdds_profiles_file)
            env_exports.extend(
                [
                    f"export FASTDDS_DEFAULT_PROFILES_FILE={quoted_profile}",
                    f"export FASTRTPS_DEFAULT_PROFILES_FILE={quoted_profile}",
                ]
            )

        env_source_label = cls._build_env_source_label(config)
        profile_source_label = "FASTDDS profile configured" if fastdds_profiles_file else "FASTDDS profile not configured"

        preflight_checks = [
            "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }",
            "ros2 interface show nav2_msgs/action/NavigateToPose >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: nav2_msgs/action/NavigateToPose is not available' >&2; exit 71; }",
            "test -n \"${ROS_DOMAIN_ID:-}\" || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ROS_DOMAIN_ID is not set' >&2; exit 72; }",
            "ros2 action list >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 action list failed' >&2; exit 73; }",
        ]

        shell_parts: list[str] = ["set -euo pipefail"]
        shell_parts.extend(env_exports)
        if remote_ros_env_cmd:
            shell_parts.append(remote_ros_env_cmd)
        elif remote_setup:
            shell_parts.append(f"source {shlex.quote(remote_setup)}")
        shell_parts.append(
            f"echo {shlex.quote(f'[transceiver] ROS env source={env_source_label}; {profile_source_label}; namespace={resolved_namespace}') }"
        )
        shell_parts.extend(preflight_checks)
        shell_parts.append(ros2_cmd)
        remote_cmd = "; ".join(shell_parts)
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={int(max(1.0, config.goal_acceptance_timeout_s))}",
            config.robot_host,
            "bash",
            "-lc",
            remote_cmd,
        ]

    def cancel_current_goal(self) -> None:
        process = self._last_process
        if process is None or process.poll() is not None:
            return

        self._cancel_on_server()
        self._wait_for_cancel_confirmation(process, timeout_s=1.0)

        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=1.5)
            return
        except Exception:
            pass

        try:
            process.terminate()
            process.wait(timeout=1.0)
            return
        except Exception:
            pass

        try:
            process.kill()
        except Exception:
            pass

    def _cancel_on_server(self) -> None:
        config = self._last_config
        goal_id = self._last_goal_id
        if config is None or not goal_id:
            return
        try:
            cancel_cmd = self._build_cancel_command(config=config, goal_id=goal_id)
            subprocess.run(
                cancel_cmd,
                capture_output=True,
                text=True,
                timeout=max(2.0, config.goal_acceptance_timeout_s),
                check=False,
            )
        except Exception:
            return

    @staticmethod
    def _wait_for_cancel_confirmation(
        process: subprocess.Popen[str],
        *,
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.05)

    @classmethod
    def _build_cancel_command(
        cls,
        *,
        config: NavigationAdapterConfig,
        goal_id: str,
    ) -> list[str]:
        resolved_action = cls._resolve_action_name(
            namespace=config.ros2_namespace,
            action_name=config.ros2_action_name,
        )
        remote_ros_env_cmd = config.remote_ros_env_cmd.strip()
        remote_setup = config.remote_ros_setup.strip()
        shell_parts: list[str] = ["set -euo pipefail"]
        if remote_ros_env_cmd:
            shell_parts.append(remote_ros_env_cmd)
        elif remote_setup:
            shell_parts.append(f"source {shlex.quote(remote_setup)}")
        shell_parts.append(
            " ".join(
                [
                    "ros2",
                    "action",
                    "cancel",
                    shlex.quote(resolved_action),
                    "--goal-id",
                    shlex.quote(goal_id),
                ]
            )
        )
        remote_cmd = "; ".join(shell_parts)
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={int(max(1.0, config.goal_acceptance_timeout_s))}",
            config.robot_host,
            "bash",
            "-lc",
            remote_cmd,
        ]

    def send_goal(
        self,
        *,
        point: NavigationPoint,
        config: NavigationAdapterConfig,
        on_feedback: Callable[[dict[str, Any]], None],
    ) -> NavigationOutcome:
        try:
            cmd = self._build_command(point, config)
        except ValueError as exc:
            return NavigationOutcome(
                terminal_state="connection_error",
                accepted=False,
                message=str(exc),
            )
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
        self._last_config = config
        self._last_goal_id = None
        accepted = False
        start = time.monotonic()
        stdout_tail: deque[str] = deque(maxlen=20)

        while True:
            line = self._last_process.stdout.readline()
            if line:
                line = line.strip()
                stdout_tail.append(line)
                lower = line.lower()
                maybe_goal_match = self._goal_id_pattern.search(lower)
                if maybe_goal_match:
                    self._last_goal_id = maybe_goal_match.group(1)
                if "goal accepted" in lower:
                    accepted = True
                elif "feedback" in lower:
                    on_feedback(
                        {
                            "raw": line,
                            "position": self._extract_position_payload(line),
                        }
                    )
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
                remaining_stdout = (
                    self._last_process.stdout.read() if self._last_process.stdout else ""
                )
                for rem_line in remaining_stdout.splitlines():
                    rem_line = rem_line.strip()
                    if rem_line:
                        stdout_tail.append(rem_line)
                stderr = self._last_process.stderr.read().strip() if self._last_process.stderr else ""
                if poll == 0:
                    if accepted:
                        return NavigationOutcome("succeeded", accepted=True)
                    return NavigationOutcome("aborted", accepted=False, message=stderr)
                stderr_tail = self._tail_text(stderr)
                stdout_summary = "\\n".join(stdout_tail)
                preflight_failed = "TRANSCEIVER_ENV_CHECK_FAILED:" in stderr
                summary = f"exit_code={poll}"
                if stdout_summary:
                    summary = f"{summary}; stdout={stdout_summary}"
                if preflight_failed:
                    reason = self._tail_text(
                        "\n".join(
                            line for line in stderr.splitlines() if "TRANSCEIVER_ENV_CHECK_FAILED:" in line
                        ),
                        max_lines=1,
                    )
                    return NavigationOutcome(
                        "connection_error",
                        accepted=False,
                        message=(
                            f"ROS environment precheck failed before sending goal: {reason}; remote_cmd={cmd[-1]}"
                        ),
                    )
                return NavigationOutcome(
                    "connection_error",
                    accepted=accepted,
                    message=(
                        f"remote command failed: {summary}; remote_cmd={cmd[-1]}; stderr={stderr_tail}"
                    ),
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


    def cancel_current_goal(self) -> None:
        self.transport.cancel_current_goal()

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
                position = payload.get("position")
                if isinstance(position, dict):
                    event_handler(NavigationEvent("position", attempt, data=position))

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
