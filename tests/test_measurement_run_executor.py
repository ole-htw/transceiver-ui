from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.measurement_run_executor import (
    JsonRunLogStore,
    MeasurementRunExecutor,
    MeasurementRunExecutorConfig,
)


class FakeNavigator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[float, float, float, float]] = []
        self.cancel_calls = 0
        self._idx = 0
        self.cancel_requested = False

    def navigate_to_point(self, point, *, timeout_s: float):
        self.calls.append((point.x, point.y, point.qz, timeout_s))
        response = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        if self.cancel_requested and response == "succeeded":
            return "connection_error"
        return response

    def cancel_current_goal(self) -> None:
        self.cancel_calls += 1
        self.cancel_requested = True


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
    assert persisted[0]["navigation"]["duration_s"] >= 0
    assert persisted[0]["measurement"]["status"] == "succeeded"
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
    class CancelAwareNavigator(FakeNavigator):
        def __init__(self) -> None:
            super().__init__(["canceled", "canceled"])
            self._released = threading.Event()

        def navigate_to_point(self, point, *, timeout_s: float):
            self.calls.append((point.x, point.y, point.qz, timeout_s))
            self._released.wait(timeout=1.0)
            return "canceled"

        def cancel_current_goal(self) -> None:
            super().cancel_current_goal()
            self._released.set()

    nav = CancelAwareNavigator()
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

    assert executor.state == "cancelled"
    assert nav.cancel_calls == 1




def test_manual_stop_during_navigation_results_in_cancelled_state() -> None:
    class BlockingNavigator(FakeNavigator):
        def __init__(self) -> None:
            super().__init__(["canceled"])
            self._released = threading.Event()

        def navigate_to_point(self, point, *, timeout_s: float):
            self.calls.append((point.x, point.y, point.qz, timeout_s))
            self._released.wait(timeout=1.0)
            return "canceled"

        def cancel_current_goal(self) -> None:
            super().cancel_current_goal()
            self._released.set()

    nav = BlockingNavigator()
    summaries: list[dict] = []
    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
        persist_run_summary=summaries.append,
    )

    thread = threading.Thread(target=executor.start)
    thread.start()
    time.sleep(0.05)
    executor.stop()
    thread.join(timeout=2)

    assert executor.state == "cancelled"
    assert summaries[0]["executor_state"] == "cancelled"
    assert summaries[0]["abort_reason"] == "run_cancelled.manual_stop"
    assert summaries[0]["completed_cycles"] == 0


def test_manual_stop_after_last_point_uses_completed_with_abort_substatus() -> None:
    class DelayedMeasurementService:
        def trigger(self, point_context):
            time.sleep(0.2)
            return {"measurement_id": point_context.global_index}

    nav = FakeNavigator(["succeeded"])
    summaries: list[dict] = []
    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="single", points=[_mission().points[0]], repeat=1),
        navigator=nav,
        measurement_service=DelayedMeasurementService(),
        persist_result=lambda _payload: None,
        persist_run_summary=summaries.append,
    )

    thread = threading.Thread(target=executor.start)
    thread.start()
    time.sleep(0.05)
    executor.stop()
    thread.join(timeout=2)

    assert executor.state == "completed"
    assert summaries[0]["completion_substatus"] == "completed_with_abort_request"
    assert summaries[0]["abort_reason"] == "run_cancelled.manual_stop_after_completion"


def test_manual_stop_without_cancel_confirmation_marks_failed() -> None:
    class NoCancelConfirmationNavigator(FakeNavigator):
        def __init__(self) -> None:
            super().__init__(["connection_error"])
            self._released = threading.Event()

        def navigate_to_point(self, point, *, timeout_s: float):
            self.calls.append((point.x, point.y, point.qz, timeout_s))
            self._released.wait(timeout=1.0)
            return "connection_error"

        def cancel_current_goal(self) -> None:
            super().cancel_current_goal()
            self._released.set()

    nav = NoCancelConfirmationNavigator()
    summaries: list[dict] = []
    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="single", points=[_mission().points[0]], repeat=1),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
        persist_run_summary=summaries.append,
        config=MeasurementRunExecutorConfig(on_point_error="stop"),
    )

    thread = threading.Thread(target=executor.start)
    thread.start()
    time.sleep(0.05)
    executor.stop()
    thread.join(timeout=2)

    assert executor.state == "failed"
    assert nav.cancel_calls == 1
    assert summaries[0]["abort_reason"] == "navigation_failed.connection_error"

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


def test_measurement_service_trigger_receives_point_context() -> None:
    nav = FakeNavigator(["succeeded"])
    observed: dict[str, object] = {}

    class Service:
        def trigger(self, point_context):
            observed["mission_name"] = point_context.mission_name
            observed["global_index"] = point_context.global_index
            observed["point_id"] = point_context.point.id
            return {"measurement_id": "m-1", "file_ref": "signals/rx/m-1.bin"}

    persisted: list[dict] = []
    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="ctx", points=[_mission().points[0]], repeat=1),
        navigator=nav,
        measurement_service=Service(),
        persist_result=persisted.append,
    )

    final_state = executor.start()

    assert final_state == "completed"
    assert observed == {"mission_name": "ctx", "global_index": 0, "point_id": "p1"}
    assert persisted[0]["measurement"]["id"] == "m-1"
    assert persisted[0]["measurement"]["file_ref"] == "signals/rx/m-1.bin"


def test_writes_point_logs_and_run_summary(tmp_path: Path) -> None:
    nav = FakeNavigator(["succeeded", "timeout"])
    run_log_store = JsonRunLogStore(tmp_path)

    executor = MeasurementRunExecutor(
        mission=_mission(),
        navigator=nav,
        trigger_measurement=lambda _point: {"measurement_id": "ok-1"},
        persist_result=lambda _payload: None,
        run_log_store=run_log_store,
        config=MeasurementRunExecutorConfig(on_point_error="continue"),
    )

    final_state = executor.start()

    assert final_state == "completed"
    point_logs = sorted(tmp_path.glob("point-*.json"))
    assert len(point_logs) == 2

    first_payload = json.loads(point_logs[0].read_text(encoding="utf-8"))
    assert first_payload["timestamps"]["start"] <= first_payload["timestamps"]["end"]
    assert first_payload["measurement"]["status"] == "succeeded"

    second_payload = json.loads(point_logs[1].read_text(encoding="utf-8"))
    assert second_payload["measurement"]["status"] == "skipped"
    assert second_payload["error"] == "navigation_failed.timeout"

    summary_payload = json.loads((tmp_path / "run-summary.json").read_text(encoding="utf-8"))
    assert summary_payload["total_points"] == 2
    assert summary_payload["succeeded_points"] == 1
    assert summary_payload["failed_points"] == 1
    assert summary_payload["skipped_points"] == 0
    assert summary_payload["completed_cycles"] == 1


def test_persist_run_summary_includes_abort_reason() -> None:
    nav = FakeNavigator(["timeout"])
    summaries: list[dict] = []

    executor = MeasurementRunExecutor(
        mission=MeasurementMission(name="summary", points=[_mission().points[0]], repeat=1),
        navigator=nav,
        trigger_measurement=lambda _point: {"ok": True},
        persist_result=lambda _payload: None,
        persist_run_summary=summaries.append,
        config=MeasurementRunExecutorConfig(on_point_error="stop"),
    )

    final_state = executor.start()

    assert final_state == "failed"
    assert len(summaries) == 1
    assert summaries[0]["executor_state"] == "failed"
    assert summaries[0]["abort_reason"] == "navigation_failed.timeout"


def test_marks_orphaned_runs_as_interrupted(tmp_path: Path) -> None:
    run_dir = tmp_path / "20240101-000000"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "run-summary.json"
    summary_path.write_text(
        json.dumps({"executor_state": "running", "abort_reason": None}),
        encoding="utf-8",
    )

    store = JsonRunLogStore(tmp_path / "20240101-000001")
    changed = store.mark_interrupted_runs()

    assert changed == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["executor_state"] == "interrupted"
    assert payload["abort_reason"] == "app_restart"
