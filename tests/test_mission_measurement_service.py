from __future__ import annotations

import subprocess
import time
from pathlib import Path

from transceiver.measurement_mission import MeasurementPoint
from transceiver.measurement_run_executor import PointExecutionContext
from transceiver.mission_measurement_service import MissionRxMeasurementService
from transceiver.navigation_adapter import Ros2CliNavigationTransport


class _FakeApp:
    def receive_for_mission(self, *, output_file: str, point_context=None):
        return {"ok": True, "output_file": output_file}


def _point_context() -> PointExecutionContext:
    return PointExecutionContext(
        mission_name="demo",
        cycle=1,
        point_index=0,
        global_index=0,
        point=MeasurementPoint(id="p-1", name="P1", x=1.0, y=2.0),
    )


def test_trigger_promotes_review_los_echo_fields_to_measurement_result() -> None:
    lidar_payload = {"topic": "/scan", "command": "ros2 topic echo /scan --once"}
    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        collect_lidar_reference=lambda _output_file: lidar_payload,
        review_measurement=lambda **_kwargs: {
            "approved": True,
            "los_lag": -120,
            "echo_indices": [33],
            "echo_lags": [-90],
            "echo_delays": [30],
        },
    )

    payload = service.trigger(_point_context())

    assert payload["los_lag"] == -120
    assert payload["echo_lags"] == [-90]
    assert payload["echo_delays"] == [
        {"echo_index": 33, "delta_lag": 30.0, "distance_m": 45.0}
    ]
    assert payload["review"]["echo_delays"] == payload["echo_delays"]
    assert payload["lidar_reference"] == lidar_payload


def test_trigger_scales_review_echo_delays_by_interpolation_factor() -> None:
    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        collect_lidar_reference=lambda _output_file: {"topic": "/scan"},
        review_measurement=lambda **_kwargs: {
            "approved": True,
            "los_lag": -120,
            "echo_indices": [33],
            "echo_lags": [-60],
            "echo_delays": [60],
            "interpolation_enabled": True,
            "interpolation_factor": 2,
        },
    )

    payload = service.trigger(_point_context())

    assert payload["echo_delays"] == [
        {"echo_index": 33, "delta_lag": 30.0, "distance_m": 45.0}
    ]


def test_trigger_waits_for_receive_and_runs_review_after_success() -> None:
    timestamps: dict[str, float] = {}

    class _SlowApp:
        def receive_for_mission(self, *, output_file: str, point_context=None):
            timestamps["receive_start"] = time.perf_counter()
            time.sleep(0.05)
            timestamps["receive_end"] = time.perf_counter()
            return {"ok": True, "output_file": output_file}

    def _review(**_kwargs):
        timestamps["review_called"] = time.perf_counter()
        return {"approved": True}

    status_events: list[tuple[str, str]] = []
    service = MissionRxMeasurementService(
        app=_SlowApp(),
        on_status=lambda phase, status: status_events.append((phase, status)),
        collect_lidar_reference=lambda _output_file: {"topic": "/scan"},
        review_measurement=_review,
    )

    started = time.perf_counter()
    payload = service.trigger(_point_context())
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.045
    assert status_events == [("measurement", "running"), ("measurement", "succeeded")]
    assert timestamps["review_called"] >= timestamps["receive_end"]
    assert payload["file_ref"].endswith(".bin")


def test_trigger_skips_lidar_reference_when_disabled() -> None:
    lidar_calls: list[str] = []

    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        collect_lidar_reference=lambda output_file: lidar_calls.append(str(output_file)) or {"topic": "/scan"},
        enable_lidar_reference=False,
    )

    payload = service.trigger(_point_context())

    assert lidar_calls == []
    assert "lidar_reference" not in payload


def test_capture_lidar_reference_uses_navigation_remote_ssh_ros_context(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs

        class _Completed:
            returncode = 0
            stdout = "scan: ok\n"
            stderr = ""

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        robot_host="robot@10.0.0.2",
        remote_ros_env_cmd="source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash",
        fastdds_profiles_file="/etc/nav2/fastdds/nav2.xml",
    )

    output_file = tmp_path / "scan.txt"
    payload = service._capture_lidar_reference(output_file)

    command = observed["command"]
    assert command[0:2] == ["ssh", "-o"]
    assert "robot@10.0.0.2" in command
    assert "BatchMode=yes" in command
    assert "ConnectTimeout=15" in command
    assert command[-3:] and command[-3] == "bash" and command[-2] == "-lc"
    remote_cmd = command[-1]
    assert "source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash" in remote_cmd
    assert "export FASTDDS_DEFAULT_PROFILES_FILE=/etc/nav2/fastdds/nav2.xml" in remote_cmd
    assert "export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/nav2/fastdds/nav2.xml" in remote_cmd
    assert "test -n \"${ROS_DOMAIN_ID:-}\"" in remote_cmd
    assert "ros2 topic info /scan >/dev/null 2>&1" in remote_cmd
    assert "ros2 topic echo /scan --once" in remote_cmd
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 15.0,
    }
    assert output_file.read_text(encoding="utf-8") == "scan: ok\n"
    assert payload["topic"] == "/scan"
    assert payload["command"].startswith("set -euo pipefail; export FASTDDS_DEFAULT_PROFILES_FILE=")


def test_capture_lidar_reference_uses_same_ssh_transport_builder_as_navigation(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        observed["command"] = command

        class _Completed:
            returncode = 0
            stdout = "scan: ok\n"
            stderr = ""

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        robot_host="robot@10.0.0.2",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
        lidar_topic="/robot1/scan",
        lidar_timeout_s=20.0,
    )

    output_file = tmp_path / "scan.txt"
    payload = service._capture_lidar_reference(output_file)

    command = observed["command"]
    assert command[0:2] == ["ssh", "-o"]
    remote_cmd = command[-1]
    assert "source /opt/ros/jazzy/setup.bash" in remote_cmd
    assert payload["topic"] == "/robot1/scan"
    assert payload["command"].endswith("ros2 topic echo /robot1/scan --once")
    assert "ros2 topic info /robot1/scan >/dev/null 2>&1" in payload["command"]

    expected = Ros2CliNavigationTransport._build_remote_ssh_command(
        robot_host="robot@10.0.0.2",
        connect_timeout_s=20.0,
        remote_ros_env_cmd="",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
        fastdds_profiles_file="",
        remote_command="ros2 topic echo /robot1/scan --once",
        diagnostics_label="lidar_topic=/robot1/scan",
        preflight_checks=[
            "echo \"TRANSCEIVER_LIDAR_DIAG whoami=$(whoami 2>/dev/null || true)\"",
            "echo \"TRANSCEIVER_LIDAR_DIAG HOME=${HOME:-}\"",
            "echo \"TRANSCEIVER_LIDAR_DIAG PWD=$(pwd 2>/dev/null || true)\"",
            "echo \"TRANSCEIVER_LIDAR_DIAG ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}\"",
            "echo \"TRANSCEIVER_LIDAR_DIAG ros2_path=$(command -v ros2 2>/dev/null || echo '<not-found>')\"",
            "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }",
            "test -n \"${ROS_DOMAIN_ID:-}\" || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ROS_DOMAIN_ID is not set' >&2; exit 72; }",
            "ros2 topic info /robot1/scan >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 topic info /robot1/scan failed' >&2; exit 74; }",
        ],
    )
    assert command == expected


def test_capture_lidar_reference_requires_robot_host(tmp_path: Path) -> None:
    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        robot_host="",
    )

    try:
        service._capture_lidar_reference(tmp_path / "scan.txt")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert exc.args == ("TRANSCEIVER_ROBOT_HOST is not configured",)


def test_trigger_reports_lidar_called_process_error_details() -> None:
    messages: list[str] = []
    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        on_operator_message=messages.append,
        collect_lidar_reference=lambda _output_file: (_ for _ in ()).throw(
            subprocess.CalledProcessError(
                returncode=1,
                cmd=["bash", "-lc", "ros2 topic echo /scan --once"],
                output="\n".join(f"line-{idx}" for idx in range(30)),
                stderr="ros2 failed hard",
            )
        ),
    )

    try:
        service.trigger(_point_context())
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert exc.args == ("lidar_reference_failed",)

    assert len(messages) == 1
    assert "returncode=1" in messages[0]
    assert "stderr=ros2 failed hard" in messages[0]
    assert "stdout_tail:\nline-10" in messages[0]
    assert "line-29" in messages[0]
    assert "line-9" not in messages[0]
