from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.measurement_run_executor import (
    JsonRunLogStore,
    MeasurementRunExecutor,
    MeasurementRunExecutorConfig,
    PointExecutionContext,
)


class ScriptedRobotAdapter:
    """Mocked robot adapter with deterministic terminal states."""

    def __init__(self, states: list[str], gate: threading.Event | None = None) -> None:
        self._states = states
        self._idx = 0
        self.calls: list[tuple[str | None, float]] = []
        self.cancel_calls = 0
        self._gate = gate

    def navigate_to_point(self, point, *, timeout_s: float):
        if self._gate is not None:
            self._gate.wait(timeout=2)
        self.calls.append((getattr(point, "x", None), timeout_s))
        state = self._states[min(self._idx, len(self._states) - 1)]
        self._idx += 1
        return state

    def cancel_current_goal(self) -> None:
        self.cancel_calls += 1


class ScriptedMeasurementService:
    def __init__(self, failures: set[int] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[PointExecutionContext] = []

    def trigger(self, point_context: PointExecutionContext) -> dict[str, str]:
        self.calls.append(point_context)
        if point_context.global_index in self.failures:
            raise RuntimeError(f"measurement_failed_at_{point_context.global_index}")
        return {"measurement_id": f"m-{point_context.global_index}"}


def _three_point_mission() -> MeasurementMission:
    return MeasurementMission(
        name="integration-mission",
        points=[
            MeasurementPoint(id="p1", name="P1", x=0.0, y=0.0, yaw=0.0),
            MeasurementPoint(id="p2", name="P2", x=1.0, y=1.0, yaw=0.0),
            MeasurementPoint(id="p3", name="P3", x=2.0, y=2.0, yaw=0.0),
        ],
    )


def test_happy_path_with_three_points_persists_logs(tmp_path: Path) -> None:
    navigator = ScriptedRobotAdapter(["succeeded", "succeeded", "succeeded"])
    measurement = ScriptedMeasurementService()
    persisted: list[dict] = []
    run_log_store = JsonRunLogStore(tmp_path)

    executor = MeasurementRunExecutor(
        mission=_three_point_mission(),
        navigator=navigator,
        measurement_service=measurement,
        persist_result=persisted.append,
        run_log_store=run_log_store,
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert [record.status for record in executor.records] == ["succeeded", "succeeded", "succeeded"]
    assert len(persisted) == 3
    assert len(list(tmp_path.glob("point-*.json"))) == 3
    summary = json.loads((tmp_path / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["succeeded_points"] == 3
    assert summary["failed_points"] == 0


def test_navigation_abort_at_second_point_continues_to_third() -> None:
    navigator = ScriptedRobotAdapter(["succeeded", "aborted", "succeeded"])
    measurement = ScriptedMeasurementService()

    executor = MeasurementRunExecutor(
        mission=_three_point_mission(),
        navigator=navigator,
        measurement_service=measurement,
        persist_result=lambda _payload: None,
        config=MeasurementRunExecutorConfig(on_point_error="continue"),
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert [record.status for record in executor.records] == ["succeeded", "failed", "succeeded"]
    assert [ctx.point.id for ctx in measurement.calls] == ["p1", "p3"]


def test_measurement_error_after_successful_navigation_continues() -> None:
    navigator = ScriptedRobotAdapter(["succeeded", "succeeded", "succeeded"])
    measurement = ScriptedMeasurementService(failures={1})

    executor = MeasurementRunExecutor(
        mission=_three_point_mission(),
        navigator=navigator,
        measurement_service=measurement,
        persist_result=lambda _payload: None,
        config=MeasurementRunExecutorConfig(on_point_error="continue"),
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert [record.status for record in executor.records] == ["succeeded", "failed", "succeeded"]
    assert executor.records[1].error == "measurement_failed_at_1"


def test_timeout_retry_and_stop_policy() -> None:
    navigator = ScriptedRobotAdapter(["timeout", "timeout", "succeeded"])

    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="single", points=[_three_point_mission().points[0]]),
        navigator=navigator,
        measurement_service=ScriptedMeasurementService(),
        persist_result=lambda _payload: None,
        config=MeasurementRunExecutorConfig(
            goal_reached_timeout_s=1.0,
            navigation_retry_attempts=1,
            on_point_error="stop",
        ),
    )

    final_state = executor.start()

    assert final_state == "failed"
    assert len(navigator.calls) == 2
    assert executor.records[0].status == "failed"
    assert executor.records[0].error == "navigation_failed:timeout"


def test_stop_and_resume_mid_run() -> None:
    gate = threading.Event()
    navigator = ScriptedRobotAdapter(["succeeded", "succeeded", "succeeded"], gate=gate)
    measurement = ScriptedMeasurementService()

    executor = MeasurementRunExecutor(
        mission=_three_point_mission(),
        navigator=navigator,
        measurement_service=measurement,
        persist_result=lambda _payload: None,
    )

    thread = threading.Thread(target=executor.start)
    thread.start()

    deadline = time.time() + 1.0
    while executor.state == "idle" and time.time() < deadline:
        time.sleep(0.01)

    executor.pause()
    assert executor.state == "paused"

    executor.resume()
    assert executor.state == "running"

    executor.stop()
    gate.set()
    thread.join(timeout=2)

    assert executor.state == "completed"
    assert navigator.cancel_calls == 1
    assert 1 <= len(executor.records) < 3
