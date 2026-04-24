from __future__ import annotations

import json
import math
import queue
import re
import shlex
import signal
import subprocess
import threading
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
    "succeeded",
    "aborted",
    "canceled",
    "timeout",
    "connection_error",
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

        quat_norm = math.sqrt(
            float(self.qx) ** 2
            + float(self.qy) ** 2
            + float(self.qz) ** 2
            + float(self.qw) ** 2
        )
        if quat_norm <= 1e-9:
            raise ValueError("orientation quaternion must not be zero")


@dataclass(frozen=True)
class NavigationAdapterConfig:
    robot_host: str = "ole@192.168.10.10"
    ros2_namespace: str = ""
    ros2_action_name: str = "/navigate_to_pose"

    # Either provide a complete shell snippet, e.g.
    # "source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
    # or a setup file path such as "/opt/ros/jazzy/setup.bash".
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

    # Rosbridge pose stream settings.
    pose_stream_ready_timeout_s: float = 2.5
    pose_stream_parent_frame_id: str = "map"
    pose_stream_child_frame_id: str = "base_footprint"
    pose_stream_topic: str = "/tf"
    pose_stream_static_topic: str = "/tf_static"
    pose_stream_throttle_rate_ms: int = 100
    pose_stream_missing_path_warn_s: float = 2.0
    pose_stream_max_dynamic_edge_age_s: float = 10.0
    pose_stream_reconnect_max_backoff_s: float = 10.0


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


class _ProcessLinePump:
    """Reads stdout/stderr without blocking the main timeout loop."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._lines: queue.Queue[tuple[str, str]] = queue.Queue()
        self._threads: list[threading.Thread] = []

        if process.stdout is not None:
            self._threads.append(
                threading.Thread(
                    target=self._read_stream,
                    args=("stdout", process.stdout),
                    daemon=True,
                )
            )
        if process.stderr is not None:
            self._threads.append(
                threading.Thread(
                    target=self._read_stream,
                    args=("stderr", process.stderr),
                    daemon=True,
                )
            )

        for thread in self._threads:
            thread.start()

    def _read_stream(self, name: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                self._lines.put((name, line))
        except Exception as exc:
            self._lines.put(("reader_error", f"{name}: {exc}"))

    def get_line(self, timeout_s: float) -> tuple[str, str] | None:
        try:
            return self._lines.get(timeout=max(0.01, timeout_s))
        except queue.Empty:
            return None

    def drain(self) -> list[tuple[str, str]]:
        drained: list[tuple[str, str]] = []
        while True:
            try:
                drained.append(self._lines.get_nowait())
            except queue.Empty:
                return drained


class Ros2CliNavigationTransport:
    """Transport via `ssh <host> ros2 action send_goal ...`.

    This keeps the public API independent from ROS Python packages. The goal is
    structured as Python data and serialized only at the SSH/CLI boundary.
    """

    action_type = "nav2_msgs/action/NavigateToPose"

    def __init__(self) -> None:
        self._last_process: subprocess.Popen[str] | None = None
        self._last_config: NavigationAdapterConfig | None = None
        self._last_goal_id: str | None = None
        self._goal_id_pattern = re.compile(
            r"(?:goal(?:[_\s-]?id)?|id)\s*[:=]\s*([0-9a-fA-F-]{36})",
            re.IGNORECASE,
        )
        self._position_block_pattern = re.compile(
            r"\bposition\b[\s:=\-\{\[]*.*?\bx\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r".*?\by\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE | re.DOTALL,
        )
        self._x_pattern = re.compile(
            r"\bx\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        self._y_pattern = re.compile(
            r"\by\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
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

    def _extract_position_payload(self, feedback_block: str) -> dict[str, Any] | None:
        match = self._position_block_pattern.search(feedback_block)
        if match:
            x_raw = match.group(1)
            y_raw = match.group(2)
        else:
            x_match = self._x_pattern.search(feedback_block)
            y_match = self._y_pattern.search(feedback_block)
            if not x_match or not y_match:
                return None
            x_raw = x_match.group(1)
            y_raw = y_match.group(1)

        try:
            x = float(x_raw)
            y = float(y_raw)
        except (TypeError, ValueError):
            return None

        yaw_match = self._yaw_pattern.search(feedback_block)
        yaw = float(yaw_match.group(1)) if yaw_match else None
        frame_match = self._frame_id_pattern.search(feedback_block)
        frame_id = frame_match.group(1).lstrip("/") if frame_match else "map"
        return {"x": x, "y": y, "yaw": yaw, "frame_id": frame_id}

    @staticmethod
    def _is_feedback_continuation_line(raw_line: str) -> bool:
        stripped = raw_line.strip()
        if not stripped:
            return False
        if raw_line[:1].isspace():
            return True
        lowered = stripped.lower()
        return any(
            token in lowered
            for token in (
                "position",
                "frame_id",
                "yaw",
                "x:",
                "y:",
                "x=",
                "y=",
                "header",
                "pose",
                "distance_remaining",
                "estimated_time_remaining",
                "number_of_recoveries",
                "navigation_time",
            )
        )

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
    def _build_remote_ssh_command(
        *,
        robot_host: str,
        connect_timeout_s: float,
        remote_ros_env_cmd: str,
        remote_ros_setup: str,
        fastdds_profiles_file: str,
        remote_command: str,
        diagnostics_label: str,
        preflight_checks: list[str] | None = None,
    ) -> list[str]:
        env_exports: list[str] = []
        if fastdds_profiles_file:
            quoted_profile = shlex.quote(fastdds_profiles_file)
            env_exports.extend(
                [
                    f"export FASTDDS_DEFAULT_PROFILES_FILE={quoted_profile}",
                    f"export FASTRTPS_DEFAULT_PROFILES_FILE={quoted_profile}",
                ]
            )

        env_source_label = "none"
        if remote_ros_env_cmd:
            env_source_label = "TRANSCEIVER_REMOTE_ROS_ENV_CMD"
        elif remote_ros_setup:
            env_source_label = "TRANSCEIVER_REMOTE_ROS_SETUP"
        profile_source_label = (
            "FASTDDS profile configured"
            if fastdds_profiles_file
            else "FASTDDS profile not configured"
        )

        shell_parts: list[str] = ["set -euo pipefail"]
        shell_parts.extend(env_exports)
        if remote_ros_env_cmd:
            shell_parts.append(remote_ros_env_cmd)
        elif remote_ros_setup:
            shell_parts.append(f"source {shlex.quote(remote_ros_setup)}")

        # Domain 0 is the ROS 2 default. Do not fail only because ROS_DOMAIN_ID
        # is absent in the remote shell.
        shell_parts.append("export ROS_DOMAIN_ID=\"${ROS_DOMAIN_ID:-0}\"")
        shell_parts.append(
            "export RMW_IMPLEMENTATION=\"${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}\""
        )
        shell_parts.append(
            f"echo {shlex.quote(f'[transceiver] ROS env source={env_source_label}; {profile_source_label}; {diagnostics_label}') }"
        )
        shell_parts.extend(preflight_checks or [])
        shell_parts.append(remote_command)
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
            f"ConnectTimeout={int(max(1.0, connect_timeout_s))}",
            robot_host,
            "bash",
            "-lc",
            remote_cmd,
        ]

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

        preflight_checks = [
            "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }",
            "ros2 interface show nav2_msgs/action/NavigateToPose >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: nav2_msgs/action/NavigateToPose is not available' >&2; exit 71; }",
            "ros2 action list >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 action list failed' >&2; exit 73; }",
        ]
        return cls._build_remote_ssh_command(
            robot_host=config.robot_host,
            connect_timeout_s=config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=config.remote_ros_env_cmd.strip(),
            remote_ros_setup=config.remote_ros_setup.strip(),
            fastdds_profiles_file=config.fastdds_profiles_file.strip(),
            remote_command=ros2_cmd,
            diagnostics_label=f"namespace={resolved_namespace}",
            preflight_checks=preflight_checks,
        )

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
        if config is None:
            return
        try:
            cancel_cmd = self._build_cancel_command(config=config, goal_id=self._last_goal_id)
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
        goal_id: str | None,
    ) -> list[str]:
        resolved_action = cls._resolve_action_name(
            namespace=config.ros2_namespace,
            action_name=config.ros2_action_name,
        )
        return cls._build_remote_ssh_command(
            robot_host=config.robot_host,
            connect_timeout_s=config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=config.remote_ros_env_cmd.strip(),
            remote_ros_setup=config.remote_ros_setup.strip(),
            fastdds_profiles_file=config.fastdds_profiles_file.strip(),
        if goal_id:
            remote_command = " ".join(
                [
                    "ros2",
                    "action",
                    "cancel",
                    shlex.quote(resolved_action),
                    "--goal-id",
                    shlex.quote(goal_id),
                ]
            )
        else:
            cancel_service_name = f"{resolved_action}/_action/cancel_goal"
            cancel_all_payload = (
                '{"goal_info":{"goal_id":{"uuid":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"stamp":{"sec":0,"nanosec":0}}}'
            )
            remote_command = " ".join(
                [
                    "ros2",
                    "service",
                    "call",
                    shlex.quote(cancel_service_name),
                    "action_msgs/srv/CancelGoal",
                    shlex.quote(cancel_all_payload),
                ]
            )
        return cls._build_remote_ssh_command(
            robot_host=config.robot_host,
            connect_timeout_s=config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=config.remote_ros_env_cmd.strip(),
            remote_ros_setup=config.remote_ros_setup.strip(),
            fastdds_profiles_file="",
            remote_command=remote_command,
            diagnostics_label=f"cancel_action={resolved_action}",
        )

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=1.0)
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
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            return NavigationOutcome(
                terminal_state="connection_error",
                accepted=False,
                message=str(exc),
            )

        self._last_process = process
        self._last_config = config
        self._last_goal_id = None

        pump = _ProcessLinePump(process)
        accepted = False
        start = time.monotonic()
        stdout_tail: deque[str] = deque(maxlen=30)
        stderr_tail: deque[str] = deque(maxlen=30)
        feedback_buffer: list[str] = []

        def _emit_feedback(feedback_lines: list[str]) -> None:
            if not feedback_lines:
                return
            feedback_block = "\n".join(feedback_lines)
            position = self._extract_position_payload(feedback_block)
            payload: dict[str, Any] = {"raw": feedback_block, "position": position}
            if position is None:
                payload["parse_error"] = "Failed to extract x/y from feedback block"
                payload["raw_feedback_excerpt"] = self._tail_text(
                    feedback_block,
                    max_lines=6,
                )
            on_feedback(payload)

        def _handle_stdout_line(raw_line: str) -> NavigationOutcome | None:
            nonlocal accepted, feedback_buffer
            line = raw_line.strip()
            if not line:
                return None

            stdout_tail.append(line)
            lower = line.lower()

            if feedback_buffer:
                if self._is_feedback_continuation_line(raw_line):
                    feedback_buffer.append(line)
                    return None
                _emit_feedback(feedback_buffer)
                feedback_buffer.clear()

            goal_match = self._goal_id_pattern.search(line)
            if goal_match:
                self._last_goal_id = goal_match.group(1)

            if "goal accepted" in lower:
                accepted = True
                return None
            if "feedback" in lower:
                feedback_buffer.append(line)
                return None
            if "succeeded" in lower:
                _emit_feedback(feedback_buffer)
                feedback_buffer.clear()
                return NavigationOutcome("succeeded", accepted=accepted)
            if "aborted" in lower:
                _emit_feedback(feedback_buffer)
                feedback_buffer.clear()
                return NavigationOutcome("aborted", accepted=accepted, message=line)
            if "canceled" in lower or "cancelled" in lower:
                _emit_feedback(feedback_buffer)
                feedback_buffer.clear()
                return NavigationOutcome("canceled", accepted=accepted, message=line)
            return None

        while True:
            item = pump.get_line(timeout_s=0.1)
            if item is not None:
                stream_name, raw_line = item
                if stream_name == "stdout":
                    maybe_outcome = _handle_stdout_line(raw_line)
                    if maybe_outcome is not None:
                        return maybe_outcome
                elif stream_name == "stderr":
                    stripped = raw_line.strip()
                    if stripped:
                        stderr_tail.append(stripped)
                elif stream_name == "reader_error":
                    stderr_tail.append(raw_line.strip())

            poll = process.poll()
            elapsed = time.monotonic() - start

            if not accepted and elapsed > config.goal_acceptance_timeout_s:
                _emit_feedback(feedback_buffer)
                if config.cancel_on_timeout:
                    self._cancel_on_server()
                self._terminate_process(process)
                return NavigationOutcome(
                    "timeout",
                    accepted=False,
                    message="Goal acceptance timeout exceeded",
                )

            if accepted and elapsed > config.goal_reached_timeout_s:
                _emit_feedback(feedback_buffer)
                if config.cancel_on_timeout:
                    self._cancel_on_server()
                self._terminate_process(process)
                return NavigationOutcome(
                    "timeout",
                    accepted=True,
                    message="Goal reached timeout exceeded",
                )

            if poll is not None:
                for stream_name, raw_line in pump.drain():
                    if stream_name == "stdout":
                        maybe_outcome = _handle_stdout_line(raw_line)
                        if maybe_outcome is not None:
                            return maybe_outcome
                    elif stream_name == "stderr":
                        stripped = raw_line.strip()
                        if stripped:
                            stderr_tail.append(stripped)

                _emit_feedback(feedback_buffer)
                feedback_buffer.clear()

                stderr = "\n".join(stderr_tail)
                stdout_summary = "\n".join(stdout_tail)

                if poll == 0:
                    if accepted:
                        return NavigationOutcome("succeeded", accepted=True)
                    return NavigationOutcome(
                        "aborted",
                        accepted=False,
                        message=stderr or stdout_summary or "ros2 action command exited without accepting goal",
                    )

                preflight_lines = [
                    line for line in stderr.splitlines() if "TRANSCEIVER_ENV_CHECK_FAILED:" in line
                ]
                if preflight_lines:
                    return NavigationOutcome(
                        "connection_error",
                        accepted=False,
                        message=(
                            "ROS environment precheck failed before sending goal: "
                            f"{preflight_lines[-1]}; remote_cmd={cmd[-1]}"
                        ),
                    )

                summary = f"exit_code={poll}"
                if stdout_summary:
                    summary = f"{summary}; stdout={stdout_summary}"
                return NavigationOutcome(
                    "connection_error",
                    accepted=accepted,
                    message=(
                        f"remote command failed: {summary}; remote_cmd={cmd[-1]}; "
                        f"stderr={self._tail_text(stderr)}"
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
            event_handler(
                NavigationEvent(
                    "goal_sent",
                    attempt,
                    data={"host": run_config.robot_host},
                )
            )

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
                # The transport already attempts server-side cancel and process cleanup.
                event_handler(
                    NavigationEvent(
                        "canceled",
                        attempt,
                        message="Canceled after timeout",
                    )
                )

            if state == "succeeded":
                return state
            if state not in run_config.retry_on_states or attempt >= attempts:
                return state

        return "aborted"


class RosbridgePoseStreamTransport:
    """Streams a TF-composed pose, usually `map -> base_footprint`, via rosbridge.

    This class intentionally keeps a small TF cache. `/tf` messages do not have
    to contain the full path in one websocket packet. For example, `map->odom`
    may arrive in one message and `odom->base_footprint` in another one.
    """

    _READY_RETRY_INTERVAL_S = 0.15

    _TfEdge = tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
        float,
        bool,
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ssh_process: subprocess.Popen[str] | None = None
        self._websocket: Any = None
        self._lock = threading.Lock()
        self._tf_edges: dict[tuple[str, str], RosbridgePoseStreamTransport._TfEdge] = {}
        self._last_missing_path_warning_monotonic = 0.0

    @staticmethod
    def _reserve_local_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _tail_text(text: str, *, max_lines: int = 6) -> str:
        return Ros2CliNavigationTransport._tail_text(text, max_lines=max_lines)

    @classmethod
    def _build_tunnel_command(cls, *, config: NavigationAdapterConfig, local_port: int) -> list[str]:
        return [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:9090",
            "-o",
            "ExitOnForwardFailure=yes",
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
        ]

    @staticmethod
    def _extract_yaw(*, x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _quat_multiply(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
            (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
            (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
        )

    @staticmethod
    def _quat_normalize(
        q: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, z, w = q
        norm = math.sqrt((x * x) + (y * y) + (z * z) + (w * w))
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm, w / norm)

    @classmethod
    def _rotate_vector_by_quaternion(
        cls,
        x: float,
        y: float,
        z: float,
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        qx, qy, qz, qw = cls._quat_normalize(q)
        vx, vy, vz = x, y, z

        ux = (qy * vz) - (qz * vy)
        uy = (qz * vx) - (qx * vz)
        uz = (qx * vy) - (qy * vx)

        uux = (qy * uz) - (qz * uy)
        uuy = (qz * ux) - (qx * uz)
        uuz = (qx * uy) - (qy * ux)

        return (
            vx + 2.0 * ((qw * ux) + uux),
            vy + 2.0 * ((qw * uy) + uuy),
            vz + 2.0 * ((qw * uz) + uuz),
        )

    @staticmethod
    def _extract_timestamp(stamp: Any) -> float:
        if isinstance(stamp, dict):
            try:
                sec = int(stamp.get("sec", 0))
                nanosec = int(stamp.get("nanosec", 0))
                return float(sec) + float(nanosec) / 1_000_000_000.0
            except (TypeError, ValueError):
                return time.time()
        return time.time()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @classmethod
    def _normalize_frame_id(cls, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lstrip("/")
        return normalized or None

    @classmethod
    def _compose_edges(
        cls,
        first_edge: _TfEdge,
        second_edge: _TfEdge,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float], float]:
        (t1x, t1y, t1z), q1, ts1, _static1 = first_edge
        (t2x, t2y, t2z), q2, ts2, _static2 = second_edge
        rt2x, rt2y, rt2z = cls._rotate_vector_by_quaternion(t2x, t2y, t2z, q1)
        px = t1x + rt2x
        py = t1y + rt2y
        pz = t1z + rt2z
        q = cls._quat_normalize(cls._quat_multiply(q1, q2))
        return (px, py, pz), q, max(ts1, ts2)

    def _prune_stale_dynamic_edges(self, *, max_age_s: float) -> None:
        if max_age_s <= 0:
            return
        now = time.time()
        stale_keys = [
            key
            for key, (_t, _q, stamp, is_static) in self._tf_edges.items()
            if not is_static and (now - stamp) > max_age_s
        ]
        for key in stale_keys:
            self._tf_edges.pop(key, None)

    def _ingest_tf_message(self, msg: dict[str, Any], *, is_static: bool) -> None:
        transforms = msg.get("transforms")
        if not isinstance(transforms, list):
            return

        for transform_entry in transforms:
            if not isinstance(transform_entry, dict):
                continue
            header = transform_entry.get("header")
            transform = transform_entry.get("transform")
            if not isinstance(header, dict) or not isinstance(transform, dict):
                continue

            parent_frame = self._normalize_frame_id(header.get("frame_id"))
            child_frame = self._normalize_frame_id(transform_entry.get("child_frame_id"))
            if parent_frame is None or child_frame is None:
                continue

            translation = transform.get("translation")
            rotation = transform.get("rotation")
            if not isinstance(translation, dict) or not isinstance(rotation, dict):
                continue

            tx = self._safe_float(translation.get("x"), 0.0)
            ty = self._safe_float(translation.get("y"), 0.0)
            tz = self._safe_float(translation.get("z"), 0.0)
            qx = self._safe_float(rotation.get("x"), 0.0)
            qy = self._safe_float(rotation.get("y"), 0.0)
            qz = self._safe_float(rotation.get("z"), 0.0)
            qw = self._safe_float(rotation.get("w"), 1.0)
            q = self._quat_normalize((qx, qy, qz, qw))

            self._tf_edges[(parent_frame, child_frame)] = (
                (tx, ty, tz),
                q,
                self._extract_timestamp(header.get("stamp")) or time.time(),
                is_static,
            )

    def _extract_pose_from_cache(
        self,
        *,
        expected_parent_frame: str,
        expected_child_frame: str,
        max_dynamic_edge_age_s: float,
    ) -> dict[str, Any] | None:
        self._prune_stale_dynamic_edges(max_age_s=max_dynamic_edge_age_s)

        direct = self._tf_edges.get((expected_parent_frame, expected_child_frame))
        if direct is not None:
            (px, py, _pz), (qx, qy, qz, qw), timestamp, _is_static = direct
            return {
                "x": px,
                "y": py,
                "frame_id": expected_parent_frame,
                "child_frame_id": expected_child_frame,
                "timestamp": timestamp,
                "yaw": self._extract_yaw(x=qx, y=qy, z=qz, w=qw),
            }

        for (parent_frame, mid_frame), first_edge in list(self._tf_edges.items()):
            if parent_frame != expected_parent_frame:
                continue
            second_edge = self._tf_edges.get((mid_frame, expected_child_frame))
            if second_edge is None:
                continue

            (px, py, _pz), (qx, qy, qz, qw), timestamp = self._compose_edges(
                first_edge,
                second_edge,
            )
            return {
                "x": px,
                "y": py,
                "frame_id": expected_parent_frame,
                "child_frame_id": expected_child_frame,
                "via_frame_id": mid_frame,
                "timestamp": timestamp,
                "yaw": self._extract_yaw(x=qx, y=qy, z=qz, w=qw),
            }

        return None

    def _extract_pose_payload(
        self,
        rosbridge_msg: dict[str, Any],
        *,
        expected_frame_id: str,
        expected_child_frame_id: str,
        max_dynamic_edge_age_s: float,
    ) -> dict[str, Any] | None:
        msg = rosbridge_msg.get("msg")
        if not isinstance(msg, dict):
            return None

        topic = rosbridge_msg.get("topic")
        is_static = topic == getattr(self, "_active_static_topic", "/tf_static")
        self._ingest_tf_message(msg, is_static=is_static)

        expected_parent_frame = self._normalize_frame_id(expected_frame_id) or "map"
        expected_child_frame = (
            self._normalize_frame_id(expected_child_frame_id) or "base_footprint"
        )
        return self._extract_pose_from_cache(
            expected_parent_frame=expected_parent_frame,
            expected_child_frame=expected_child_frame,
            max_dynamic_edge_age_s=max_dynamic_edge_age_s,
        )

    def _stop_ssh_tunnel(self) -> None:
        with self._lock:
            process = self._ssh_process
            self._ssh_process = None
        if process is None:
            return
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

    def _close_websocket(self) -> None:
        with self._lock:
            ws = self._websocket
            self._websocket = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            pass

    def start(
        self,
        *,
        config: NavigationAdapterConfig,
        on_event: Callable[[dict[str, Any]], None],
        expected_frame_id: str | None = None,
        expected_child_frame_id: str | None = None,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        with self._lock:
            self._tf_edges.clear()
        self._last_missing_path_warning_monotonic = 0.0
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={
                "config": config,
                "on_event": on_event,
                "expected_frame_id": expected_frame_id,
                "expected_child_frame_id": expected_child_frame_id,
            },
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_websocket()
        self._stop_ssh_tunnel()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def _emit_missing_path_warning_if_due(
        self,
        *,
        on_event: Callable[[dict[str, Any]], None],
        reconnect_attempt: int,
        expected_frame: str,
        expected_child_frame: str,
        warn_interval_s: float,
    ) -> None:
        now = time.monotonic()
        if now - self._last_missing_path_warning_monotonic < max(0.25, warn_interval_s):
            return
        self._last_missing_path_warning_monotonic = now
        known_edges = [f"{parent}->{child}" for parent, child in sorted(self._tf_edges.keys())]
        on_event(
            {
                "type": "pose_stream",
                "event": {
                    "type": "stream_warning",
                    "message": (
                        "tf rosbridge stream has not produced expected transform path "
                        f"({expected_frame}->{expected_child_frame} directly or via one intermediate frame)"
                    ),
                    "known_edges": known_edges[-20:],
                    "attempt": reconnect_attempt,
                    "timestamp": time.time(),
                },
            }
        )

    def _run_loop(
        self,
        *,
        config: NavigationAdapterConfig,
        on_event: Callable[[dict[str, Any]], None],
        expected_frame_id: str | None = None,
        expected_child_frame_id: str | None = None,
    ) -> None:
        try:
            import websocket  # type: ignore
        except ImportError:
            on_event(
                {
                    "type": "pose_stream",
                    "event": {
                        "type": "stream_error",
                        "message": "python package 'websocket-client' is required for rosbridge pose streaming",
                        "attempt": 1,
                        "timestamp": time.time(),
                    },
                }
            )
            return

        expected_frame = (
            expected_frame_id.strip()
            if isinstance(expected_frame_id, str) and expected_frame_id.strip()
            else config.pose_stream_parent_frame_id
        ).strip().lstrip("/") or "map"
        expected_child_frame = (
            expected_child_frame_id.strip()
            if isinstance(expected_child_frame_id, str) and expected_child_frame_id.strip()
            else config.pose_stream_child_frame_id
        ).strip().lstrip("/") or "base_footprint"

        pose_topic = config.pose_stream_topic.strip() or "/tf"
        static_topic = config.pose_stream_static_topic.strip() or "/tf_static"
        self._active_static_topic = static_topic

        reconnect_attempt = 0
        while not self._stop_event.is_set():
            reconnect_attempt += 1
            disconnect_reason = ""
            try:
                local_port = self._reserve_local_port()
                tunnel_cmd = self._build_tunnel_command(config=config, local_port=local_port)
                process = subprocess.Popen(
                    tunnel_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                with self._lock:
                    self._ssh_process = process

                ready_timeout_s = max(0.1, float(config.pose_stream_ready_timeout_s))
                ready_deadline = time.monotonic() + ready_timeout_s
                ws = None
                last_connect_error: Exception | None = None

                while time.monotonic() < ready_deadline and not self._stop_event.is_set():
                    poll_code = process.poll()
                    if poll_code is not None:
                        stderr = process.stderr.read() if process.stderr is not None else ""
                        disconnect_reason = (
                            f"SSH tunnel setup failed (exit_code={poll_code}): "
                            f"{self._tail_text(stderr, max_lines=3)}"
                        )
                        raise OSError(disconnect_reason)
                    try:
                        ws = websocket.create_connection(
                            f"ws://127.0.0.1:{local_port}",
                            timeout=2.0,
                        )
                        break
                    except Exception as exc:
                        last_connect_error = exc
                        self._stop_event.wait(timeout=self._READY_RETRY_INTERVAL_S)

                if ws is None:
                    poll_code = process.poll()
                    if poll_code is not None:
                        stderr = process.stderr.read() if process.stderr is not None else ""
                        disconnect_reason = (
                            f"SSH tunnel setup failed (exit_code={poll_code}): "
                            f"{self._tail_text(stderr, max_lines=3)}"
                        )
                    else:
                        disconnect_reason = f"tunnel not ready within {ready_timeout_s:.1f}s"
                        if last_connect_error is not None:
                            disconnect_reason = (
                                f"{disconnect_reason} (last connect error: {last_connect_error})"
                            )
                    raise OSError(disconnect_reason)

                ws.settimeout(0.5)
                with self._lock:
                    self._websocket = ws

                ws.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "topic": pose_topic,
                            "type": "tf2_msgs/msg/TFMessage",
                            "throttle_rate": max(0, int(config.pose_stream_throttle_rate_ms)),
                        }
                    )
                )
                ws.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "topic": static_topic,
                            "type": "tf2_msgs/msg/TFMessage",
                        }
                    )
                )
                on_event(
                    {
                        "type": "pose_stream",
                        "event": {
                            "type": "stream_connected",
                            "attempt": reconnect_attempt,
                            "timestamp": time.time(),
                            "local_port": local_port,
                            "pose_topic": pose_topic,
                            "static_topic": static_topic,
                            "expected_frame_id": expected_frame,
                            "expected_child_frame_id": expected_child_frame,
                        },
                    }
                )

                while not self._stop_event.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception as exc:
                        disconnect_reason = f"rosbridge websocket disconnected: {exc}"
                        break

                    if not isinstance(raw, str) or not raw.strip():
                        continue

                    try:
                        decoded = json.loads(raw)
                    except json.JSONDecodeError:
                        on_event(
                            {
                                "type": "pose_stream",
                                "event": {
                                    "type": "stream_error",
                                    "message": "rosbridge message is not valid JSON",
                                    "raw_pose_excerpt": self._tail_text(raw, max_lines=2),
                                    "attempt": reconnect_attempt,
                                    "timestamp": time.time(),
                                },
                            }
                        )
                        continue

                    if decoded.get("op") != "publish":
                        continue
                    if decoded.get("topic") not in {pose_topic, static_topic}:
                        continue

                    payload = self._extract_pose_payload(
                        decoded,
                        expected_frame_id=expected_frame,
                        expected_child_frame_id=expected_child_frame,
                        max_dynamic_edge_age_s=max(
                            0.0,
                            float(config.pose_stream_max_dynamic_edge_age_s),
                        ),
                    )
                    if payload is None:
                        self._emit_missing_path_warning_if_due(
                            on_event=on_event,
                            reconnect_attempt=reconnect_attempt,
                            expected_frame=expected_frame,
                            expected_child_frame=expected_child_frame,
                            warn_interval_s=float(config.pose_stream_missing_path_warn_s),
                        )
                        continue

                    on_event(
                        {
                            "type": "pose_stream",
                            "event": {
                                "type": "position_update",
                                "position": payload,
                                "timestamp": time.time(),
                            },
                        }
                    )

                if not disconnect_reason and not self._stop_event.is_set():
                    disconnect_reason = "pose stream disconnected"
            except OSError as exc:
                disconnect_reason = str(exc)
            except Exception as exc:
                disconnect_reason = f"pose stream connection failed: {exc}"
            finally:
                self._close_websocket()
                self._stop_ssh_tunnel()

            if self._stop_event.is_set():
                break

            on_event(
                {
                    "type": "pose_stream",
                    "event": {
                        "type": "stream_error",
                        "message": disconnect_reason or "pose stream disconnected after connection",
                        "attempt": reconnect_attempt,
                        "timestamp": time.time(),
                    },
                }
            )
            backoff_s = min(
                max(0.1, float(config.pose_stream_reconnect_max_backoff_s)),
                float(2 ** min(reconnect_attempt - 1, 4)),
            )
            on_event(
                {
                    "type": "pose_stream",
                    "event": {
                        "type": "stream_reconnect_wait",
                        "backoff_s": backoff_s,
                        "attempt": reconnect_attempt,
                        "timestamp": time.time(),
                    },
                }
            )
            self._stop_event.wait(timeout=backoff_s)


Ros2CliPoseStreamTransport = RosbridgePoseStreamTransport

