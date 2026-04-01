from __future__ import annotations

import time
from pathlib import Path

from transceiver.measurement_mission import MeasurementPoint
from transceiver.measurement_run_executor import PointExecutionContext
from transceiver.mission_measurement_service import MissionRxMeasurementService


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
        {"echo_index": 33, "delta_lag": 30, "distance_m": 45.0}
    ]
    assert payload["review"]["echo_delays"] == payload["echo_delays"]
    assert payload["lidar_reference"] == lidar_payload


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


def test_capture_lidar_reference_uses_bash_shell_with_configured_ros_env(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs

        class _Completed:
            stdout = "scan: ok\n"

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        lidar_ros_env_cmd="source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash",
    )

    output_file = tmp_path / "scan.txt"
    payload = service._capture_lidar_reference(output_file)

    assert observed["command"] == [
        "bash",
        "-lc",
        "set -euo pipefail; source /opt/ros/jazzy/setup.bash && source ~/ws/install/setup.bash; "
        "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }; "
        "ros2 topic echo /scan --once",
    ]
    assert observed["kwargs"] == {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": 15.0,
    }
    assert output_file.read_text(encoding="utf-8") == "scan: ok\n"
    assert payload["topic"] == "/scan"
    assert payload["command"].startswith("set -euo pipefail; source /opt/ros/jazzy/setup.bash")


def test_capture_lidar_reference_prefers_setup_file_when_ros_env_cmd_is_empty(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        observed["command"] = command

        class _Completed:
            stdout = "scan: ok\n"

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
        lidar_ros_setup="/opt/ros/humble/setup.bash",
        lidar_topic="/robot1/scan",
        lidar_timeout_s=20.0,
    )

    output_file = tmp_path / "scan.txt"
    payload = service._capture_lidar_reference(output_file)

    assert observed["command"] == [
        "bash",
        "-lc",
        "set -euo pipefail; source /opt/ros/humble/setup.bash; "
        "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }; "
        "ros2 topic echo /robot1/scan --once",
    ]
    assert payload["topic"] == "/robot1/scan"
    assert payload["command"].endswith("ros2 topic echo /robot1/scan --once")
