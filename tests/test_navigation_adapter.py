from __future__ import annotations

from dataclasses import dataclass
import signal

from transceiver.navigation_adapter import (
    NavigationAdapter,
    NavigationAdapterConfig,
    NavigationOutcome,
    NavigationPoint,
    Ros2CliNavigationTransport,
    RosbridgePoseStreamTransport,
)


@dataclass
class _FakeTransport:
    outcomes: list[NavigationOutcome]

    def __post_init__(self) -> None:
        self.send_calls = 0
        self.cancel_calls = 0
        self.server_cancel_confirmed = False

    def send_goal(self, *, point, config, on_feedback):
        self.send_calls += 1
        on_feedback({"remaining_distance": 1.23})
        idx = min(self.send_calls - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[idx]
        if self.cancel_calls and self.server_cancel_confirmed and outcome.terminal_state != "canceled":
            return NavigationOutcome("canceled", accepted=True, message="server cancel confirmed")
        return outcome

    def cancel_current_goal(self) -> None:
        self.cancel_calls += 1


def test_build_goal_payload_uses_navigate_to_pose_in_map_frame() -> None:
    payload = Ros2CliNavigationTransport.build_goal_payload(
        NavigationPoint(x=1.0, y=2.0, z=0.5, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
    )

    assert payload["pose"]["header"]["frame_id"] == "map"
    assert payload["pose"]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.5}
    assert payload["pose"]["pose"]["orientation"] == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "w": 0.9,
    }


def test_adapter_retries_on_connection_error_and_emits_events() -> None:
    fake = _FakeTransport(
        outcomes=[
            NavigationOutcome("connection_error", accepted=False, message="ssh failed"),
            NavigationOutcome("succeeded", accepted=True, message="done"),
        ]
    )
    adapter = NavigationAdapter(
        transport=fake,
        config=NavigationAdapterConfig(retry_attempts=1),
    )

    events: list[str] = []
    state = adapter.navigate_to_point(
        NavigationPoint(0.0, 0.0),
        on_event=lambda event: events.append(event.type),
    )

    assert state == "succeeded"
    assert fake.send_calls == 2
    assert events.count("goal_sent") == 2
    assert "connection_error" in events
    assert "feedback" in events
    assert "accepted" in events
    assert "succeeded" in events


def test_timeout_can_trigger_cancel_event() -> None:
    fake = _FakeTransport(outcomes=[NavigationOutcome("timeout", accepted=True)])
    adapter = NavigationAdapter(
        transport=fake,
        config=NavigationAdapterConfig(retry_attempts=0, cancel_on_timeout=True),
    )

    events: list[str] = []
    state = adapter.navigate_to_point(
        NavigationPoint(3.0, 4.0),
        on_event=lambda event: events.append(event.type),
    )

    assert state == "timeout"
    assert fake.cancel_calls == 1
    assert "timeout" in events
    assert "canceled" in events


def test_build_command_uses_non_interactive_ssh_options() -> None:
    config = NavigationAdapterConfig(robot_host="robot@10.0.0.2", ros2_namespace="robot1")

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    assert cmd[0:2] == ["ssh", "-o"]
    assert "BatchMode=yes" in cmd
    assert "PasswordAuthentication=no" in cmd
    assert "KbdInteractiveAuthentication=no" in cmd
    assert "NumberOfPasswordPrompts=0" in cmd
    assert "StrictHostKeyChecking=accept-new" in cmd
    assert "robot@10.0.0.2" in cmd


def test_build_command_uses_bash_lc_and_sources_ros_setup_when_configured() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        ros2_namespace="robot1",
        ros2_action_name="navigate_to_pose",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    assert cmd[-3] == "bash"
    assert cmd[-2] == "-lc"
    remote_cmd = cmd[-1]
    assert remote_cmd.startswith("set -euo pipefail; source /opt/ros/jazzy/setup.bash; echo '[transceiver] ROS env source=TRANSCEIVER_REMOTE_ROS_SETUP; FASTDDS profile not configured; namespace=robot1'; command -v ros2")
    assert "/robot1/navigate_to_pose" in remote_cmd


def test_build_command_uses_default_ros_command_without_setup() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        ros2_namespace="robot1",
        ros2_action_name="navigate_to_pose",
        remote_ros_setup="",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    assert cmd[-3] == "bash"
    assert cmd[-2] == "-lc"
    remote_cmd = cmd[-1]
    assert remote_cmd.startswith("set -euo pipefail; echo '[transceiver] ROS env source=none; FASTDDS profile not configured; namespace=robot1'; command -v ros2")
    assert "source " not in remote_cmd
    assert "'" in remote_cmd
    assert "\"frame_id\":\"map\"" in remote_cmd




def test_build_command_exports_fastdds_profile_env_vars() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        fastdds_profiles_file="/etc/nav2/fastdds/nav2.xml",
        ros2_namespace="robot1",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    remote_cmd = cmd[-1]
    assert "export FASTDDS_DEFAULT_PROFILES_FILE=/etc/nav2/fastdds/nav2.xml" in remote_cmd
    assert "export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nav2/fastdds/nav2.xml" in remote_cmd
    assert "ros2 action send_goal" in remote_cmd


def test_build_command_prefers_explicit_remote_ros_env_cmd() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        remote_ros_env_cmd="source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
        ros2_namespace="robot1",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    assert cmd[-3] == "bash"
    assert cmd[-2] == "-lc"
    remote_cmd = cmd[-1]
    assert remote_cmd.startswith(
        "set -euo pipefail; source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash; echo '[transceiver] ROS env source=TRANSCEIVER_REMOTE_ROS_ENV_CMD; FASTDDS profile not configured; namespace=robot1'; command -v ros2"
    )


def test_build_command_with_ros_setup_wraps_command_with_pipefail_and_send_goal() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
        ros2_namespace="robot1",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    remote_cmd = cmd[-1]
    assert "set -euo pipefail;" in remote_cmd
    assert "source /opt/ros/jazzy/setup.bash;" in remote_cmd
    assert "echo '[transceiver] ROS env source=TRANSCEIVER_REMOTE_ROS_SETUP; FASTDDS profile not configured; namespace=robot1'" in remote_cmd
    assert "ros2 interface show nav2_msgs/action/NavigateToPose" in remote_cmd
    assert "test -n \"${ROS_DOMAIN_ID:-}\"" in remote_cmd
    assert "ros2 action list >/dev/null 2>&1" in remote_cmd
    assert "ros2 action send_goal" in remote_cmd


def test_send_goal_formats_remote_command_failure_with_truncated_output(monkeypatch) -> None:
    class _Stream:
        def __init__(self, lines: list[str], rest: str = "") -> None:
            self._lines = lines
            self._rest = rest
            self._idx = 0

        def readline(self) -> str:
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        def read(self) -> str:
            return self._rest

    class _Process:
        def __init__(self) -> None:
            self.stdout = _Stream(
                ["line-1\n", "line-2\n"],
                rest="\n".join(f"line-{i}" for i in range(3, 28)),
            )
            self.stderr = _Stream([], rest="\n".join(f"err-{i}" for i in range(1, 28)))

        def poll(self):
            return 42

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    transport = Ros2CliNavigationTransport()
    outcome = transport.send_goal(
        point=NavigationPoint(1.0, 2.0),
        config=NavigationAdapterConfig(robot_host="robot@10.0.0.2", remote_ros_setup="/opt/ros/jazzy/setup.bash", ros2_namespace="robot1"),
        on_feedback=lambda _feedback: None,
    )

    assert outcome.terminal_state == "connection_error"
    assert outcome.message is not None
    assert outcome.message.startswith("remote command failed: exit_code=42; stdout=")
    assert "remote_cmd=set -euo pipefail; source /opt/ros/jazzy/setup.bash; echo '[transceiver] ROS env source=TRANSCEIVER_REMOTE_ROS_SETUP; FASTDDS profile not configured; namespace=robot1'; command -v ros2" in outcome.message
    assert "\nline-1\n" not in f"\n{outcome.message}\n"
    assert "line-8" in outcome.message
    assert "line-27" in outcome.message
    assert "\nerr-1\n" not in f"\n{outcome.message}\n"
    assert "err-8" in outcome.message
    assert "err-27" in outcome.message


def test_build_command_allows_empty_namespace() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        ros2_namespace="",
        ros2_action_name="navigate_to_pose",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    remote_cmd = cmd[-1]
    assert "namespace='; command -v ros2" in remote_cmd
    assert "ros2 action send_goal /navigate_to_pose" in remote_cmd


def test_send_goal_returns_precheck_error_message(monkeypatch) -> None:
    class _Stream:
        def __init__(self, lines: list[str], rest: str = "") -> None:
            self._lines = lines
            self._rest = rest
            self._idx = 0

        def readline(self) -> str:
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        def read(self) -> str:
            return self._rest

    class _Process:
        def __init__(self) -> None:
            self.stdout = _Stream([], rest="")
            self.stderr = _Stream([], rest="TRANSCEIVER_ENV_CHECK_FAILED: ROS_DOMAIN_ID is not set")

        def poll(self):
            return 72

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    transport = Ros2CliNavigationTransport()
    outcome = transport.send_goal(
        point=NavigationPoint(1.0, 2.0),
        config=NavigationAdapterConfig(
            robot_host="robot@10.0.0.2",
            remote_ros_setup="/opt/ros/jazzy/setup.bash",
            ros2_namespace="robot1",
        ),
        on_feedback=lambda _feedback: None,
    )

    assert outcome.terminal_state == "connection_error"
    assert outcome.message is not None
    assert outcome.message.startswith("ROS environment precheck failed before sending goal")
    assert "ROS_DOMAIN_ID is not set" in outcome.message


def test_cancel_current_goal_prefers_sigint_before_force_stopping() -> None:
    class _Process:
        def __init__(self) -> None:
            self.sent_signals: list[int] = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return None

        def send_signal(self, sig: int) -> None:
            self.sent_signals.append(sig)

        def wait(self, timeout: float) -> int:
            return 0

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

    transport = Ros2CliNavigationTransport()
    process = _Process()
    transport._last_process = process

    transport.cancel_current_goal()

    assert process.sent_signals == [signal.SIGINT]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_cancel_current_goal_falls_back_to_terminate_and_kill() -> None:
    class _Process:
        def __init__(self) -> None:
            self.sent_signals: list[int] = []
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        def poll(self):
            return None

        def send_signal(self, sig: int) -> None:
            self.sent_signals.append(sig)
            raise TimeoutError("sigint failed")

        def wait(self, timeout: float) -> int:
            self.wait_calls += 1
            raise TimeoutError("still running")

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

    transport = Ros2CliNavigationTransport()
    process = _Process()
    transport._last_process = process

    transport.cancel_current_goal()

    assert process.sent_signals == [signal.SIGINT]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_cancel_current_goal_uses_server_side_cancel_before_signals(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _Process:
        def __init__(self) -> None:
            self.sent_signals: list[int] = []

        def poll(self):
            return None

        def send_signal(self, sig: int) -> None:
            self.sent_signals.append(sig)
            calls.append(("signal", sig))

        def wait(self, timeout: float) -> int:
            raise TimeoutError("still running")

        def terminate(self) -> None:
            calls.append(("terminate", None))

        def kill(self) -> None:
            calls.append(("kill", None))

    def _fake_run(cmd, **kwargs):
        calls.append(("server_cancel", cmd))
        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.run", _fake_run)

    transport = Ros2CliNavigationTransport()
    process = _Process()
    transport._last_process = process
    transport._last_config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        ros2_namespace="robot1",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
    )
    transport._last_goal_id = "00000000-0000-0000-0000-000000000001"

    transport.cancel_current_goal()

    assert calls[0][0] == "server_cancel"
    cancel_cmd = " ".join(str(part) for part in calls[0][1])
    assert "--goal-id" in cancel_cmd
    assert "00000000-0000-0000-0000-000000000001" in cancel_cmd
    assert process.sent_signals == [signal.SIGINT]


def test_send_goal_maps_dead_transport_to_connection_error_even_after_accept(monkeypatch) -> None:
    class _Stream:
        def __init__(self, lines: list[str], rest: str = "") -> None:
            self._lines = lines
            self._rest = rest
            self._idx = 0

        def readline(self) -> str:
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        def read(self) -> str:
            return self._rest

    class _Process:
        def __init__(self) -> None:
            self.stdout = _Stream(
                [
                    "Goal accepted with ID: 00000000-0000-0000-0000-000000000001\n",
                    "feedback: moving\n",
                ]
            )
            self.stderr = _Stream([], rest="ssh channel lost")

        def poll(self):
            return 255

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    transport = Ros2CliNavigationTransport()
    outcome = transport.send_goal(
        point=NavigationPoint(1.0, 2.0),
        config=NavigationAdapterConfig(
            robot_host="robot@10.0.0.2",
            remote_ros_setup="/opt/ros/jazzy/setup.bash",
            ros2_namespace="robot1",
        ),
        on_feedback=lambda _feedback: None,
    )

    assert outcome.accepted is True
    assert outcome.terminal_state == "connection_error"
    assert transport._last_goal_id == "00000000-0000-0000-0000-000000000001"


def test_extract_position_payload_parses_multiline_feedback_block() -> None:
    transport = Ros2CliNavigationTransport()
    payload = transport._extract_position_payload(
        "\n".join(
            [
                "feedback:",
                "  current_pose:",
                "    header:",
                "      frame_id: odom",
                "    pose:",
                "      position:",
                "        x: 1.25",
                "        y: -3.5",
                "  yaw: 0.75",
            ]
        )
    )

    assert payload == {"x": 1.25, "y": -3.5, "yaw": 0.75, "frame_id": "odom"}


def test_send_goal_emits_parse_diagnostics_when_feedback_position_parse_fails(monkeypatch) -> None:
    class _Stream:
        def __init__(self, lines: list[str], rest: str = "") -> None:
            self._lines = lines
            self._rest = rest
            self._idx = 0

        def readline(self) -> str:
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        def read(self) -> str:
            return self._rest

    class _Process:
        def __init__(self) -> None:
            self.stdout = _Stream(
                [
                    "Goal accepted with ID: 00000000-0000-0000-0000-000000000001\n",
                    "feedback:\n",
                    "  status: moving\n",
                    "  details: no coordinate keys in this block\n",
                    "Goal succeeded\n",
                ]
            )
            self.stderr = _Stream([], rest="")

        def poll(self):
            return None

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    feedback_payloads: list[dict[str, object]] = []
    transport = Ros2CliNavigationTransport()
    outcome = transport.send_goal(
        point=NavigationPoint(1.0, 2.0),
        config=NavigationAdapterConfig(
            robot_host="robot@10.0.0.2",
            remote_ros_setup="/opt/ros/jazzy/setup.bash",
            ros2_namespace="robot1",
        ),
        on_feedback=lambda payload: feedback_payloads.append(payload),
    )

    assert outcome.terminal_state == "succeeded"
    assert len(feedback_payloads) == 1
    payload = feedback_payloads[0]
    assert payload["position"] is None
    assert payload["parse_error"] == "Failed to extract x/y from feedback block"
    assert "status: moving" in str(payload["raw_feedback_excerpt"])


def test_rosbridge_pose_stream_build_tunnel_command() -> None:
    config = NavigationAdapterConfig(robot_host="robot@10.0.0.2")

    cmd = RosbridgePoseStreamTransport._build_tunnel_command(config=config, local_port=19090)

    assert cmd[:4] == ["ssh", "-N", "-L", "19090:127.0.0.1:9090"]
    assert "BatchMode=yes" in cmd
    assert "StrictHostKeyChecking=accept-new" in cmd
    assert "robot@10.0.0.2" in cmd


def test_rosbridge_pose_stream_sends_subscribe_and_parses_pose(monkeypatch) -> None:
    import sys
    import types

    sent_messages: list[str] = []

    class _FakeWebSocket:
        def __init__(self) -> None:
            self._recv_calls = 0

        def settimeout(self, _timeout: float) -> None:
            return None

        def send(self, message: str) -> None:
            sent_messages.append(message)

        def recv(self) -> str:
            self._recv_calls += 1
            if self._recv_calls == 1:
                return (
                    '{"op":"publish","topic":"/tf","msg":'
                    '{"transforms":[{"header":{"frame_id":"map","stamp":{"sec":10,"nanosec":500000000}},'
                    '"child_frame_id":"base_footprint","transform":{"translation":{"x":1.25,"y":2.5},'
                    '"rotation":{"x":0.0,"y":0.0,"z":0.7071068,"w":0.7071068}}}]}}'
                )
            raise RuntimeError("disconnect")

        def close(self) -> None:
            return None

    class _FakeWebsocketModule(types.SimpleNamespace):
        class WebSocketTimeoutException(Exception):
            pass

    fake_module = _FakeWebsocketModule()
    fake_module.create_connection = lambda *_args, **_kwargs: _FakeWebSocket()
    monkeypatch.setitem(sys.modules, "websocket", fake_module)

    class _Process:
        def __init__(self) -> None:
            self.stdout = None
            self.stderr = types.SimpleNamespace(read=lambda: "")
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    transport = RosbridgePoseStreamTransport()
    events: list[dict[str, object]] = []

    def _on_event(payload: dict[str, object]) -> None:
        events.append(payload)
        event = payload.get("event") if isinstance(payload, dict) else None
        if isinstance(event, dict) and event.get("type") == "position_update":
            transport._stop_event.set()

    transport._run_loop(config=NavigationAdapterConfig(robot_host="robot@10.0.0.2"), on_event=_on_event)

    assert sent_messages
    subscribe = sent_messages[0]
    assert '"op": "subscribe"' in subscribe
    assert '"topic": "/tf"' in subscribe
    updates = [
        payload["event"]["position"]
        for payload in events
        if isinstance(payload.get("event"), dict) and payload["event"].get("type") == "position_update"
    ]
    assert updates
    assert updates[0]["x"] == 1.25
    assert updates[0]["y"] == 2.5
    assert updates[0]["frame_id"] == "map"
    assert updates[0]["child_frame_id"] == "base_footprint"
    assert abs(float(updates[0]["timestamp"]) - 10.5) < 1e-6
    assert abs(float(updates[0]["yaw"]) - 1.57079632679) < 1e-3


def test_rosbridge_pose_stream_reconnects_after_disconnect(monkeypatch) -> None:
    import sys
    import types

    connection_counter = {"count": 0}

    class _FakeWebSocket:
        def settimeout(self, _timeout: float) -> None:
            return None

        def send(self, _message: str) -> None:
            return None

        def recv(self) -> str:
            raise RuntimeError("socket down")

        def close(self) -> None:
            return None

    class _FakeWebsocketModule(types.SimpleNamespace):
        class WebSocketTimeoutException(Exception):
            pass

    def _create_connection(*_args, **_kwargs):
        connection_counter["count"] += 1
        return _FakeWebSocket()

    fake_module = _FakeWebsocketModule(create_connection=_create_connection)
    monkeypatch.setitem(sys.modules, "websocket", fake_module)

    class _Process:
        def __init__(self) -> None:
            self.stderr = types.SimpleNamespace(read=lambda: "")

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Process())

    transport = RosbridgePoseStreamTransport()
    events: list[dict[str, object]] = []

    def _on_event(payload: dict[str, object]) -> None:
        events.append(payload)
        event = payload.get("event") if isinstance(payload, dict) else None
        if isinstance(event, dict) and event.get("type") == "stream_reconnect_wait":
            transport._stop_event.set()

    monkeypatch.setattr(transport._stop_event, "wait", lambda timeout=0.0: True)

    transport._run_loop(config=NavigationAdapterConfig(robot_host="robot@10.0.0.2"), on_event=_on_event)

    assert connection_counter["count"] >= 1
    assert any(
        isinstance(payload.get("event"), dict) and payload["event"].get("type") == "stream_reconnect_wait"
        for payload in events
    )


def test_rosbridge_pose_stream_stop_closes_websocket_and_tunnel(monkeypatch) -> None:
    import types

    tunnel_processes: list[object] = []

    class _Process:
        def __init__(self) -> None:
            self.stderr = types.SimpleNamespace(read=lambda: "")
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            return 0

    class _Ws:
        def __init__(self) -> None:
            self.closed = False

        def settimeout(self, _timeout: float) -> None:
            return None

        def send(self, _message: str) -> None:
            return None

        def recv(self) -> str:
            raise RuntimeError("drop")

        def close(self) -> None:
            self.closed = True

    def _popen(*_a, **_k):
        proc = _Process()
        tunnel_processes.append(proc)
        return proc

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", _popen)

    transport = RosbridgePoseStreamTransport()
    ws = _Ws()
    transport._websocket = ws
    proc = _Process()
    transport._ssh_process = proc

    transport.stop()

    assert ws.closed is True
    assert proc.terminated is True
