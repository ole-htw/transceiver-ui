from __future__ import annotations

import json
import math
import re
import select
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
    pose_stream_ready_timeout_s: float = 2.5


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
        frame_id = frame_match.group(1) if frame_match else "map"
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
    def _build_env_source_label(config: NavigationAdapterConfig) -> str:
        if config.remote_ros_env_cmd.strip():
            return "TRANSCEIVER_REMOTE_ROS_ENV_CMD"
        if config.remote_ros_setup.strip():
            return "TRANSCEIVER_REMOTE_ROS_SETUP"
        return "none"

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
        remote_ros_env_cmd = config.remote_ros_env_cmd.strip()
        remote_setup = config.remote_ros_setup.strip()
        fastdds_profiles_file = config.fastdds_profiles_file.strip()

        preflight_checks = [
            "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }",
            "ros2 interface show nav2_msgs/action/NavigateToPose >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: nav2_msgs/action/NavigateToPose is not available' >&2; exit 71; }",
            "test -n \"${ROS_DOMAIN_ID:-}\" || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ROS_DOMAIN_ID is not set' >&2; exit 72; }",
            "ros2 action list >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 action list failed' >&2; exit 73; }",
        ]
        return cls._build_remote_ssh_command(
            robot_host=config.robot_host,
            connect_timeout_s=config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=remote_ros_env_cmd,
            remote_ros_setup=remote_setup,
            fastdds_profiles_file=fastdds_profiles_file,
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
        return cls._build_remote_ssh_command(
            robot_host=config.robot_host,
            connect_timeout_s=config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=config.remote_ros_env_cmd.strip(),
            remote_ros_setup=config.remote_ros_setup.strip(),
            fastdds_profiles_file="",
            remote_command=" ".join(
                [
                    "ros2",
                    "action",
                    "cancel",
                    shlex.quote(resolved_action),
                    "--goal-id",
                    shlex.quote(goal_id),
                ]
            ),
            diagnostics_label=f"cancel_action={resolved_action}",
        )

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
        feedback_buffer: list[str] = []

        def _emit_feedback(feedback_lines: list[str]) -> None:
            if not feedback_lines:
                return
            feedback_block = "\n".join(feedback_lines)
            position = self._extract_position_payload(feedback_block)
            payload: dict[str, Any] = {"raw": feedback_block, "position": position}
            if position is None:
                payload["parse_error"] = "Failed to extract x/y from feedback block"
                payload["raw_feedback_excerpt"] = self._tail_text(feedback_block, max_lines=6)
            on_feedback(payload)

        while True:
            raw_line = self._last_process.stdout.readline()
            if raw_line:
                line = raw_line.strip()
                stdout_tail.append(line)
                lower = line.lower()
                if feedback_buffer:
                    if self._is_feedback_continuation_line(raw_line):
                        feedback_buffer.append(line)
                        continue
                    _emit_feedback(feedback_buffer)
                    feedback_buffer.clear()
                maybe_goal_match = self._goal_id_pattern.search(lower)
                if maybe_goal_match:
                    self._last_goal_id = maybe_goal_match.group(1)
                if "goal accepted" in lower:
                    accepted = True
                elif "feedback" in lower:
                    feedback_buffer.append(line)
                elif "succeeded" in lower:
                    _emit_feedback(feedback_buffer)
                    return NavigationOutcome("succeeded", accepted=accepted)
                elif "aborted" in lower:
                    _emit_feedback(feedback_buffer)
                    return NavigationOutcome("aborted", accepted=accepted, message=line)
                elif "canceled" in lower or "cancelled" in lower:
                    _emit_feedback(feedback_buffer)
                    return NavigationOutcome("canceled", accepted=accepted, message=line)

            poll = self._last_process.poll()
            elapsed = time.monotonic() - start
            if not accepted and elapsed > config.goal_acceptance_timeout_s:
                _emit_feedback(feedback_buffer)
                return NavigationOutcome(
                    "timeout",
                    accepted=False,
                    message="Goal acceptance timeout exceeded",
                )
            if accepted and elapsed > config.goal_reached_timeout_s:
                _emit_feedback(feedback_buffer)
                return NavigationOutcome(
                    "timeout",
                    accepted=True,
                    message="Goal reached timeout exceeded",
                )
            if poll is not None:
                _emit_feedback(feedback_buffer)
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


class RosbridgePoseStreamTransport:
    """Continuously streams `/base_footprint` updates via rosbridge through an SSH tunnel."""

    _POSE_TOPIC = "/base_footprint"
    _POSE_TYPE = "geometry_msgs/msg/PoseWithCovarianceStamped"
    _DEFAULT_EXPECTED_FRAME_ID = "map"
    _READY_RETRY_INTERVAL_S = 0.15

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ssh_process: subprocess.Popen[str] | None = None
        self._websocket: Any = None
        self._lock = threading.Lock()

    @staticmethod
    def _reserve_local_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @classmethod
    def _build_tunnel_command(cls, *, config: NavigationAdapterConfig, local_port: int) -> list[str]:
        return [
            "ssh",
            "-N",
            "-L",
            f"{local_port}:127.0.0.1:9090",
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
    def _tail_text(text: str, *, max_lines: int = 6) -> str:
        return Ros2CliNavigationTransport._tail_text(text, max_lines=max_lines)

    @staticmethod
    def _extract_yaw(*, x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @classmethod
    def _extract_pose_payload(cls, rosbridge_msg: dict[str, Any]) -> dict[str, Any] | None:
        msg = rosbridge_msg.get("msg")
        if not isinstance(msg, dict):
            return None
        header = msg.get("header")
        pose_cov = msg.get("pose")
        if not isinstance(header, dict) or not isinstance(pose_cov, dict):
            return None
        pose = pose_cov.get("pose")
        if not isinstance(pose, dict):
            return None
        position = pose.get("position")
        orientation = pose.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return None
        try:
            px = float(position["x"])
            py = float(position["y"])
            qx = float(orientation.get("x", 0.0))
            qy = float(orientation.get("y", 0.0))
            qz = float(orientation.get("z", 0.0))
            qw = float(orientation.get("w", 1.0))
        except (TypeError, ValueError, KeyError):
            return None

        frame_id_raw = header.get("frame_id")
        frame_id = frame_id_raw if isinstance(frame_id_raw, str) and frame_id_raw.strip() else "map"
        stamp = header.get("stamp")
        timestamp = time.time()
        if isinstance(stamp, dict):
            try:
                sec = int(stamp.get("sec", 0))
                nanosec = int(stamp.get("nanosec", 0))
                timestamp = float(sec) + float(nanosec) / 1_000_000_000.0
            except (TypeError, ValueError):
                timestamp = time.time()

        return {
            "x": px,
            "y": py,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "yaw": cls._extract_yaw(x=qx, y=qy, z=qz, w=qw),
        }

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
    ) -> None:
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={
                "config": config,
                "on_event": on_event,
                "expected_frame_id": expected_frame_id,
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

    def _run_loop(
        self,
        *,
        config: NavigationAdapterConfig,
        on_event: Callable[[dict[str, Any]], None],
        expected_frame_id: str | None = None,
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
            else self._DEFAULT_EXPECTED_FRAME_ID
        )
        reconnect_attempt = 0
        while not self._stop_event.is_set():
            reconnect_attempt += 1
            frame_mismatch_reported = False
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
                            f"SSH tunnel setup failed (exit_code={poll_code}): {self._tail_text(stderr, max_lines=3)}"
                        )
                        raise OSError(disconnect_reason)
                    try:
                        ws = websocket.create_connection(f"ws://127.0.0.1:{local_port}", timeout=2.0)
                        break
                    except Exception as exc:
                        last_connect_error = exc
                        self._stop_event.wait(timeout=self._READY_RETRY_INTERVAL_S)

                if ws is None:
                    poll_code = process.poll()
                    if poll_code is not None:
                        stderr = process.stderr.read() if process.stderr is not None else ""
                        disconnect_reason = (
                            f"SSH tunnel setup failed (exit_code={poll_code}): {self._tail_text(stderr, max_lines=3)}"
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
                            "topic": self._POSE_TOPIC,
                            "type": self._POSE_TYPE,
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
                    if decoded.get("op") != "publish" or decoded.get("topic") != self._POSE_TOPIC:
                        continue
                    payload = self._extract_pose_payload(decoded)
                    if payload is None:
                        on_event(
                            {
                                "type": "pose_stream",
                                "event": {
                                    "type": "stream_error",
                                    "message": "base_footprint rosbridge message is missing required pose fields",
                                    "attempt": reconnect_attempt,
                                    "timestamp": time.time(),
                                },
                            }
                        )
                        continue
                    received_frame = payload.get("frame_id")
                    if (
                        not frame_mismatch_reported
                        and isinstance(received_frame, str)
                        and received_frame.strip()
                        and received_frame != expected_frame
                    ):
                        frame_mismatch_reported = True
                        on_event(
                            {
                                "type": "pose_stream",
                                "event": {
                                    "type": "stream_error",
                                    "message": (
                                        "unerwarteter /base_footprint frame_id: "
                                        f"erwartet={expected_frame}, empfangen={received_frame}"
                                    ),
                                    "attempt": reconnect_attempt,
                                    "timestamp": time.time(),
                                },
                            }
                        )
                    on_event(
                        {
                            "type": "pose_stream",
                            "event": {
                                "type": "position_update",
                                "position": payload,
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
            backoff_s = min(10.0, float(2 ** min(reconnect_attempt - 1, 4)))
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
