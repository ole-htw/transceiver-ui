from __future__ import annotations

from dataclasses import dataclass

from transceiver.navigation_adapter import (
    NavigationAdapter,
    NavigationAdapterConfig,
    NavigationOutcome,
    NavigationPoint,
    Ros2CliNavigationTransport,
)


@dataclass
class _FakeTransport:
    outcomes: list[NavigationOutcome]

    def __post_init__(self) -> None:
        self.send_calls = 0
        self.cancel_calls = 0

    def send_goal(self, *, point, config, on_feedback):
        self.send_calls += 1
        on_feedback({"remaining_distance": 1.23})
        idx = min(self.send_calls - 1, len(self.outcomes) - 1)
        return self.outcomes[idx]

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
    config = NavigationAdapterConfig(robot_host="robot@10.0.0.2")

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
        remote_ros_setup="/opt/ros/humble/setup.bash",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    assert cmd[-3] == "bash"
    assert cmd[-2] == "-lc"
    remote_cmd = cmd[-1]
    assert "source /opt/ros/humble/setup.bash && ros2 action send_goal" in remote_cmd
    assert "/robot1/navigate_to_pose" in remote_cmd




def test_build_command_uses_remote_ros_env_cmd_with_pipefail_prefix() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        ros2_namespace="robot1",
        ros2_action_name="navigate_to_pose",
        remote_ros_env_cmd="source /opt/ros/humble/setup.bash",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    remote_cmd = cmd[-1]
    assert "set -euo pipefail; source /opt/ros/humble/setup.bash; ros2 action send_goal" in remote_cmd
    assert "/robot1/navigate_to_pose" in remote_cmd


def test_build_command_prefers_remote_ros_env_cmd_over_remote_ros_setup() -> None:
    config = NavigationAdapterConfig(
        robot_host="robot@10.0.0.2",
        remote_ros_env_cmd="source /env/cmd.bash",
        remote_ros_setup="/opt/ros/humble/setup.bash",
    )

    cmd = Ros2CliNavigationTransport._build_command(NavigationPoint(1.0, 2.0), config)

    remote_cmd = cmd[-1]
    assert "set -euo pipefail; source /env/cmd.bash; ros2 action send_goal" in remote_cmd
    assert "source /opt/ros/humble/setup.bash &&" not in remote_cmd
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
    assert remote_cmd.startswith("ros2 action send_goal")
    assert "source " not in remote_cmd
    assert "'" in remote_cmd
    assert "\"frame_id\":\"map\"" in remote_cmd


def test_send_goal_reports_structured_remote_failure_message(monkeypatch) -> None:
    class _Stdout:
        def __init__(self) -> None:
            self._lines = ["goal accepted\n"]

        def readline(self) -> str:
            return self._lines.pop(0) if self._lines else ""

        def read(self) -> str:
            return "out-1\nout-2\n"

    class _Stderr:
        def read(self) -> str:
            return "err-1\nerr-2\n"

    class _Proc:
        def __init__(self) -> None:
            self.stdout = _Stdout()
            self.stderr = _Stderr()

        def poll(self) -> int | None:
            if self.stdout._lines:
                return None
            return 7

    monkeypatch.setattr("transceiver.navigation_adapter.subprocess.Popen", lambda *a, **k: _Proc())

    transport = Ros2CliNavigationTransport()
    outcome = transport.send_goal(
        point=NavigationPoint(0.0, 0.0),
        config=NavigationAdapterConfig(robot_host="robot@10.0.0.2"),
        on_feedback=lambda _payload: None,
    )

    assert outcome.terminal_state == "aborted"
    assert outcome.message is not None
    assert "remote command failed:" in outcome.message
    assert "remote_cmd=ros2 action send_goal" in outcome.message
    assert "stderr=err-1" in outcome.message
    assert "stdout=goal accepted" in outcome.message
