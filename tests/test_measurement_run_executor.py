from __future__ import annotations

import threading
import time

import pytest

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.measurement_run_executor import (
    MeasurementRunExecutor,
    MeasurementRunExecutorConfig,
)


class FakeNavigator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[float, float, float, float]] = []
        self._idx = 0

    def navigate_to_point(self, point, *, timeout_s: float):
        self.calls.append((point.x, point.y, point.qz, timeout_s))
        response = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        return response


def _mission() -> MeasurementMission:
    return MeasurementMission(
        name="m1",
        points=[
            MeasurementPoint(id="p1", name="A", x=1.0, y=2.0, yaw=0.0),
            MeasurementPoint(id="p2", name="B", x=3.0, y=4.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        ],
        repeat=1,
    )


def test_executes_points_strictly_in_order_and_persists_context() -> None:
    nav = FakeNavigator(["succeeded", "succeeded"])
    measured: list[str] = []
    persisted: list[dict] = []

    def trigger(point: MeasurementPoint) -> dict:
        measured.append(point.id or "")
        return {"value": point.x + point.y}

    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=nav,
        trigger_measurement=trigger,
        persist_result=persisted.append,
        config=MeasurementRunExecutorConfig(goal_reached_timeout_s=42.0),
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert measured == ["p1", "p2"]
    assert [rec.point_id for rec in executor.records] == ["p1", "p2"]
    assert all(rec.status == "succeeded" for rec in executor.records)
    assert [item[0:2] for item in nav.calls] == [(1.0, 2.0), (3.0, 4.0)]
    assert all(item[3] == 42.0 for item in nav.calls)
    assert len(persisted) == 2
    assert persisted[0]["point"]["id"] == "p1"
    assert persisted[1]["point"]["id"] == "p2"


def test_navigation_failure_retries_then_continue() -> None:
    nav = FakeNavigator(["timeout", "aborted", "succeeded"])
    persisted: list[dict] = []

    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="m2", points=[_mission().points[0]], repeat=1),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=persisted.append,
        config=MeasurementRunExecutorConfig(navigation_retry_attempts=2, on_point_error="continue"),
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert len(nav.calls) == 3
    assert len(executor.records) == 1
    assert executor.records[0].status == "succeeded"
    assert len(persisted) == 1


def test_navigation_failure_retries_then_stop_mission() -> None:
    nav = FakeNavigator(["timeout", "aborted"])

    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
        config=MeasurementRunExecutorConfig(navigation_retry_attempts=1, on_point_error="stop"),
    )

    final_state = executor.start()

    assert final_state == "failed"
    assert len(executor.records) == 1
    assert executor.records[0].status == "failed"


def test_state_transitions_start_pause_resume_stop() -> None:
    class SlowNavigator(FakeNavigator):
        def navigate_to_point(self, point, *, timeout_s: float):
            time.sleep(0.15)
            return super().navigate_to_point(point, timeout_s=timeout_s)

    nav = SlowNavigator(["succeeded", "succeeded"])
    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
    )

    thread = threading.Thread(target=executor.start)
    thread.start()

    time.sleep(0.02)
    executor.pause()
    assert executor.state == "paused"

    executor.resume()
    assert executor.state == "running"

    executor.stop()
    thread.join(timeout=2)

    assert executor.state == "completed"


def test_invalid_transitions_raise() -> None:
    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=FakeNavigator(["succeeded", "succeeded"]),
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
    )

    with pytest.raises(RuntimeError):
        executor.pause()
    with pytest.raises(RuntimeError):
        executor.resume()
    with pytest.raises(RuntimeError):
        executor.stop()
