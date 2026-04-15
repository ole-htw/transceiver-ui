from __future__ import annotations

import math
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.navigation_adapter import NavigationPoint, TerminalNavigationState

ExecutorState = Literal[
    "idle",
    "running",
    "paused",
    "stopping",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]
OnPointError = Literal["continue", "stop"]
PointExecutionStatus = Literal["succeeded", "failed", "skipped"]


class MissionNavigator(Protocol):
    def navigate_to_point(
        self,
        point: NavigationPoint,
        *,
        timeout_s: float,
        on_navigation_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> TerminalNavigationState:
        ...

    def cancel_current_goal(self) -> None:
        ...


class MeasurementTrigger(Protocol):
    def __call__(self, point: MeasurementPoint) -> dict[str, Any]:
        ...


class ResultStore(Protocol):
    def __call__(self, payload: dict[str, Any]) -> None:
        ...


class MeasurementService(Protocol):
    def trigger(self, point_context: "PointExecutionContext") -> dict[str, Any]:
        ...


class RunSummaryStore(Protocol):
    def __call__(self, payload: dict[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class PointExecutionContext:
    mission_name: str
    cycle: int
    point_index: int
    global_index: int
    point: MeasurementPoint


class CallableMeasurementService:
    """Adapter to wrap a plain callable into MeasurementService."""

    def __init__(self, trigger_measurement: MeasurementTrigger) -> None:
        self._trigger_measurement = trigger_measurement

    def trigger(self, point_context: PointExecutionContext) -> dict[str, Any]:
        return self._trigger_measurement(point_context.point)


class JsonRunLogStore:
    """Writes one point log per measurement point plus run-summary.json."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def mark_interrupted_runs(self) -> int:
        interrupted = 0
        for summary_path in self.directory.parent.glob("*/run-summary.json"):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("executor_state") in {"running", "paused", "stopping"}:
                payload["executor_state"] = "interrupted"
                payload["abort_reason"] = payload.get("abort_reason") or "app_restart"
                summary_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                interrupted += 1
        return interrupted

    def write_point_log(self, payload: dict[str, Any]) -> None:
        point = payload["point"]
        global_index = payload.get("global_index", 0)
        point_id = point.get("id") or point.get("name") or f"point-{global_index}"
        filename = f"point-{global_index:04d}-{_sanitize_filename(str(point_id))}.json"
        (self.directory / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_run_summary(self, payload: dict[str, Any]) -> None:
        (self.directory / "run-summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-") or "point"


@dataclass(frozen=True)
class MeasurementRunExecutorConfig:
    goal_reached_timeout_s: float = 120.0
    navigation_retry_attempts: int = 0
    on_point_error: OnPointError = "continue"
    start_point_index: int = 0
    reverse_point_order: bool = False
    enable_measurements: bool = True


@dataclass(frozen=True)
class PointExecutionRecord:
    index: int
    point_id: str | None
    point_name: str | None
    status: PointExecutionStatus
    navigation_state: TerminalNavigationState | None
    navigation_attempts: int
    measurement_result: dict[str, Any] | None
    error: str | None
    timestamp: float = field(default_factory=time.time)


class MeasurementRunExecutor:
    """Executes mission points strictly in sequence.

    Workflow per point:
      1) start navigation
      2) wait for goal reached (with timeout)
      3) on success trigger measurement
      4) persist measurement result + context
      5) continue with next point
    """

    def __init__(
        self,
        *,
        mission: MeasurementMission,
        navigator: MissionNavigator,
        measurement_service: MeasurementService | None = None,
        trigger_measurement: MeasurementTrigger | None = None,
        persist_result: ResultStore,
        persist_run_summary: RunSummaryStore | None = None,
        run_log_store: JsonRunLogStore | None = None,
        on_runtime_event: Callable[[dict[str, Any]], None] | None = None,
        config: MeasurementRunExecutorConfig | None = None,
    ) -> None:
        self.mission = mission
        self.navigator = navigator
        self.measurement_service = self._resolve_measurement_service(
            measurement_service=measurement_service,
            trigger_measurement=trigger_measurement,
        )
        self.persist_result = persist_result
        self.persist_run_summary = persist_run_summary
        self.run_log_store = run_log_store
        self.on_runtime_event = on_runtime_event
        self.config = config or MeasurementRunExecutorConfig()
        self._validate_start_point_index()

        self._state: ExecutorState = "idle"
        self._state_lock = threading.RLock()
        self._pause_cond = threading.Condition(self._state_lock)
        self.records: list[PointExecutionRecord] = []
        self._cancel_requested = False
        self._cancel_confirmed = False

    @property
    def state(self) -> ExecutorState:
        with self._state_lock:
            return self._state

    def start(self) -> ExecutorState:
        abort_reason: str | None = None
        completion_substatus: str | None = None
        active_points = self._ordered_active_points()
        points_per_cycle = len(active_points)
        repeats = self.mission.repeat or 1
        expected_points = max(0, points_per_cycle * repeats - self.config.start_point_index)
        with self._state_lock:
            if self._state not in {"idle", "completed", "failed", "cancelled", "interrupted"}:
                raise RuntimeError(f"Cannot start from state '{self._state}'")
            if points_per_cycle == 0:
                self._state = "failed"
                abort_reason = "mission_has_no_active_points"
                self._write_run_summary(
                    abort_reason=abort_reason,
                    completion_substatus=completion_substatus,
                    expected_points=0,
                    points_per_cycle=0,
                )
                return self._state
            self._state = "running"

        self.records = []
        self._cancel_requested = False
        self._cancel_confirmed = False
        for cycle in range(repeats):
            for point_index, point in enumerate(active_points):
                if cycle == 0 and point_index < self.config.start_point_index:
                    continue
                if not self._wait_until_resumed_or_stopped():
                    abort_reason = "run_cancelled.manual_stop"
                    final_state = self._finalize_stop()
                    self._write_run_summary(
                        abort_reason=abort_reason,
                        completion_substatus=completion_substatus,
                        expected_points=expected_points,
                        points_per_cycle=points_per_cycle,
                    )
                    return final_state

                if self.state == "stopping":
                    abort_reason = "run_cancelled.manual_stop"
                    final_state = self._finalize_stop()
                    self._write_run_summary(
                        abort_reason=abort_reason,
                        completion_substatus=completion_substatus,
                        expected_points=expected_points,
                        points_per_cycle=points_per_cycle,
                    )
                    return final_state

                record = self._execute_point(
                    point=point,
                    point_index=point_index,
                    cycle=cycle,
                    points_per_cycle=points_per_cycle,
                )
                self.records.append(record)

                if record.status == "failed" and self.config.on_point_error == "stop":
                    with self._state_lock:
                        self._state = "failed"
                    abort_reason = record.error
                    self._write_run_summary(
                        abort_reason=abort_reason,
                        completion_substatus=completion_substatus,
                        expected_points=expected_points,
                        points_per_cycle=points_per_cycle,
                    )
                    return self.state

                if self.mission.wait_after_arrival_s > 0 and self.state == "running":
                    time.sleep(self.mission.wait_after_arrival_s)

        with self._state_lock:
            if self._state == "stopping":
                if len(self.records) >= expected_points:
                    self._state = "completed"
                    completion_substatus = "completed_with_abort_request"
                    abort_reason = "run_cancelled.manual_stop_after_completion"
                else:
                    if self._cancel_confirmed:
                        abort_reason = "run_cancelled.manual_stop"
                        self._state = "cancelled"
                    else:
                        abort_reason = "run_cancelled.manual_stop_not_confirmed"
                        self._state = "failed"
            elif self._cancel_requested:
                self._state = "completed"
                completion_substatus = "completed_with_abort_request"
                abort_reason = "run_cancelled.manual_stop_after_completion"
            else:
                self._state = "completed"
        self._write_run_summary(
            abort_reason=abort_reason,
            completion_substatus=completion_substatus,
            expected_points=expected_points,
            points_per_cycle=points_per_cycle,
        )
        return self.state

    @staticmethod
    def _resolve_measurement_service(
        *,
        measurement_service: MeasurementService | None,
        trigger_measurement: MeasurementTrigger | None,
    ) -> MeasurementService:
        if measurement_service is not None and trigger_measurement is not None:
            raise ValueError(
                "Provide either 'measurement_service' or 'trigger_measurement', not both"
            )
        if measurement_service is not None:
            return measurement_service
        if trigger_measurement is None:
            raise ValueError(
                "Missing measurement integration. Provide 'measurement_service' or 'trigger_measurement'."
            )
        return CallableMeasurementService(trigger_measurement)

    def pause(self) -> None:
        with self._state_lock:
            if self._state != "running":
                raise RuntimeError(f"Cannot pause from state '{self._state}'")
            self._state = "paused"

    def resume(self) -> None:
        with self._state_lock:
            if self._state != "paused":
                raise RuntimeError(f"Cannot resume from state '{self._state}'")
            self._state = "running"
            self._pause_cond.notify_all()

    def stop(self) -> None:
        with self._state_lock:
            if self._state not in {"running", "paused"}:
                raise RuntimeError(f"Cannot stop from state '{self._state}'")
            self._cancel_requested = True
            self._state = "stopping"
            self._pause_cond.notify_all()

        def _cancel_goal() -> None:
            try:
                self.navigator.cancel_current_goal()
            except Exception:
                pass

        threading.Thread(target=_cancel_goal, daemon=True).start()

    def _finalize_stop(self) -> ExecutorState:
        with self._state_lock:
            self._state = "cancelled" if self._cancel_confirmed else "failed"
            return self._state

    def _wait_until_resumed_or_stopped(self) -> bool:
        with self._state_lock:
            while self._state == "paused":
                self._pause_cond.wait(timeout=0.1)
            return self._state != "stopping"

    def _execute_point(
        self,
        *,
        point: MeasurementPoint,
        point_index: int,
        cycle: int,
        points_per_cycle: int,
    ) -> PointExecutionRecord:
        nav_state: TerminalNavigationState | None = None
        attempts = self.config.navigation_retry_attempts + 1
        point_started_at = time.time()
        global_index = cycle * points_per_cycle + point_index

        def _emit_navigation_event(event_payload: dict[str, Any]) -> None:
            self._emit_runtime_event(
                {
                    "type": "navigation",
                    "cycle": cycle,
                    "point_index": point_index,
                    "global_index": global_index,
                    "event": event_payload,
                    "timestamp": time.time(),
                }
            )

        for attempt in range(1, attempts + 1):
            nav_state = self.navigator.navigate_to_point(
                self._to_navigation_point(point),
                timeout_s=self.config.goal_reached_timeout_s,
                on_navigation_event=_emit_navigation_event,
            )
            if nav_state == "succeeded":
                break

        if self._cancel_requested and nav_state == "succeeded":
            nav_state = "canceled"
        if self._cancel_requested and nav_state == "canceled":
            self._cancel_confirmed = True

        if nav_state != "succeeded":
            error = (
                "run_cancelled.manual_stop"
                if self._cancel_requested and nav_state == "canceled"
                else f"navigation_failed.{nav_state}"
            )
            status: PointExecutionStatus = "skipped" if self._cancel_requested and nav_state == "canceled" else "failed"
            measurement_status = "skipped"
            
            record = PointExecutionRecord(
                index=global_index,
                point_id=point.id,
                point_name=point.name,
                status=status,
                navigation_state=nav_state,
                navigation_attempts=attempts,
                measurement_result=None,
                error=error,
            )
            self._persist_point_log(
                point=point,
                point_index=point_index,
                cycle=cycle,
                global_index=global_index,
                navigation_state=nav_state,
                navigation_attempts=attempts,
                point_started_at=point_started_at,
                measurement_result=None,
                measurement_status=measurement_status,
                error=record.error,
                navigation_duration_s=max(0.0, time.time() - point_started_at),
            )
            return record

        navigation_done_at = time.time()
        if self._cancel_requested:
            record = PointExecutionRecord(
                index=global_index,
                point_id=point.id,
                point_name=point.name,
                status="skipped",
                navigation_state="canceled",
                navigation_attempts=attempt,
                measurement_result=None,
                error="run_cancelled.manual_stop",
            )
            self._persist_point_log(
                point=point,
                point_index=point_index,
                cycle=cycle,
                global_index=global_index,
                navigation_state="canceled",
                navigation_attempts=attempt,
                point_started_at=point_started_at,
                measurement_result=None,
                measurement_status="skipped",
                error=record.error,
                navigation_duration_s=max(0.0, navigation_done_at - point_started_at),
            )
            return record

        if not self.config.enable_measurements:
            self._persist_point_log(
                point=point,
                point_index=point_index,
                cycle=cycle,
                global_index=global_index,
                navigation_state=nav_state,
                navigation_attempts=attempt,
                point_started_at=point_started_at,
                measurement_result={"mode": "test_run"},
                measurement_status="skipped",
                error=None,
                navigation_duration_s=max(0.0, navigation_done_at - point_started_at),
            )
            return PointExecutionRecord(
                index=global_index,
                point_id=point.id,
                point_name=point.name,
                status="succeeded",
                navigation_state=nav_state,
                navigation_attempts=attempt,
                measurement_result={"mode": "test_run"},
                error=None,
            )

        try:
            measurement_result = self.measurement_service.trigger(
                PointExecutionContext(
                    mission_name=self.mission.name,
                    cycle=cycle,
                    point_index=point_index,
                    global_index=global_index,
                    point=point,
                )
            )
        except Exception as exc:
            error_code = str(exc)
            review_reason: str | None = None
            review_detail: str | None = None
            if isinstance(exc, RuntimeError) and exc.args:
                first = exc.args[0]
                if isinstance(first, str) and first.strip():
                    error_code = first.strip()
                if len(exc.args) > 1 and isinstance(exc.args[1], str) and exc.args[1].strip():
                    review_reason = exc.args[1].strip()
                if len(exc.args) > 2 and isinstance(exc.args[2], str) and exc.args[2].strip():
                    review_detail = exc.args[2].strip()
            failed_measurement_payload: dict[str, Any] | None = None
            if review_reason is not None or review_detail is not None:
                failed_measurement_payload = {
                    "review": {
                        "approved": False,
                        "reason": review_reason,
                        "detail": review_detail,
                    }
                }
            record = PointExecutionRecord(
                index=global_index,
                point_id=point.id,
                point_name=point.name,
                status="failed",
                navigation_state=nav_state,
                navigation_attempts=attempt,
                measurement_result=None,
                error=error_code,
            )
            self._persist_point_log(
                point=point,
                point_index=point_index,
                cycle=cycle,
                global_index=global_index,
                navigation_state=nav_state,
                navigation_attempts=attempt,
                point_started_at=point_started_at,
                measurement_result=failed_measurement_payload,
                measurement_status="failed",
                error=record.error,
                navigation_duration_s=max(0.0, navigation_done_at - point_started_at),
            )
            return record

        self._persist_point_log(
            point=point,
            point_index=point_index,
            cycle=cycle,
            global_index=global_index,
            navigation_state=nav_state,
            navigation_attempts=attempt,
            point_started_at=point_started_at,
            measurement_result=measurement_result,
            measurement_status="succeeded",
            error=None,
            navigation_duration_s=max(0.0, navigation_done_at - point_started_at),
        )

        return PointExecutionRecord(
            index=global_index,
            point_id=point.id,
            point_name=point.name,
            status="succeeded",
            navigation_state=nav_state,
            navigation_attempts=attempt,
            measurement_result=measurement_result,
            error=None,
        )

    def _emit_runtime_event(self, payload: dict[str, Any]) -> None:
        if self.on_runtime_event is None:
            return
        try:
            self.on_runtime_event(payload)
        except Exception:
            return

    def _persist_point_log(
        self,
        *,
        point: MeasurementPoint,
        point_index: int,
        cycle: int,
        global_index: int,
        navigation_state: TerminalNavigationState | None,
        navigation_attempts: int,
        navigation_duration_s: float,
        point_started_at: float,
        measurement_result: dict[str, Any] | None,
        measurement_status: str,
        error: str | None,
    ) -> None:
        point_ended_at = time.time()
        measurement_payload = deepcopy(measurement_result)
        payload = {
            "mission": self.mission.name,
            "cycle": cycle,
            "global_index": global_index,
            "point_index": point_index,
            "point": {
                "id": point.id,
                "name": point.name,
                "target": {
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                },
                "orientation": {
                    "yaw": point.yaw,
                    "qx": point.qx,
                    "qy": point.qy,
                    "qz": point.qz,
                    "qw": point.qw,
                },
                "notes": point.notes,
                "measurement_profile": point.measurement_profile,
                "enabled": point.enabled,
            },
            "timestamps": {
                "start": point_started_at,
                "end": point_ended_at,
            },
            "navigation": {
                "state": navigation_state,
                "attempts": navigation_attempts,
                "timeout_s": self.config.goal_reached_timeout_s,
                "duration_s": navigation_duration_s,
            },
            "measurement": {
                "status": measurement_status,
                "id": _extract_measurement_id(measurement_payload),
                "file_ref": _extract_measurement_file_ref(measurement_payload),
                "result": measurement_payload,
            },
            "error": error,
            "executor_state": self.state,
        }
        self.persist_result(payload)
        if self.run_log_store is not None:
            self.run_log_store.write_point_log(payload)

    def _write_run_summary(
        self,
        *,
        abort_reason: str | None,
        completion_substatus: str | None,
        expected_points: int,
        points_per_cycle: int,
    ) -> None:
        total = len(self.records)
        succeeded = sum(1 for record in self.records if record.status == "succeeded")
        failed = sum(1 for record in self.records if record.status == "failed")
        skipped = sum(1 for record in self.records if record.status == "skipped")
        completed_cycles = 0
        if points_per_cycle > 0 and self.state == "completed":
            completed_cycles = min(total // points_per_cycle, self.mission.repeat or 1)
        summary = {
            "mission": self.mission.name,
            "executor_state": self.state,
            "total_points": total,
            "expected_points": expected_points,
            "start_point_index": self.config.start_point_index,
            "reverse_point_order": self.config.reverse_point_order,
            "succeeded_points": succeeded,
            "failed_points": failed,
            "skipped_points": skipped,
            "completed_cycles": completed_cycles,
            "abort_reason": abort_reason,
            "completion_substatus": completion_substatus,
        }
        if self.persist_run_summary is not None:
            self.persist_run_summary(summary)
        if self.run_log_store is not None:
            self.run_log_store.write_run_summary(summary)

    def _validate_start_point_index(self) -> None:
        if self.config.start_point_index < 0:
            raise ValueError("start_point_index must be >= 0")
        active_points = self._ordered_active_points()
        if active_points and self.config.start_point_index >= len(active_points):
            raise ValueError(
                "start_point_index must be smaller than number of active mission points"
            )
        if not active_points and self.config.start_point_index != 0:
            raise ValueError("start_point_index must be 0 when no active mission points exist")

    def _ordered_active_points(self) -> list[MeasurementPoint]:
        active_points = [point for point in self.mission.points if point.enabled]
        if self.config.reverse_point_order:
            return list(reversed(active_points))
        return active_points

    @staticmethod
    def _to_navigation_point(point: MeasurementPoint) -> NavigationPoint:
        if point.yaw is not None:
            return NavigationPoint(
                x=point.x,
                y=point.y,
                z=point.z,
                qx=0.0,
                qy=0.0,
                qz=math.sin(point.yaw / 2.0),
                qw=math.cos(point.yaw / 2.0),
            )
        return NavigationPoint(
            x=point.x,
            y=point.y,
            z=point.z,
            qx=point.qx if point.qx is not None else 0.0,
            qy=point.qy if point.qy is not None else 0.0,
            qz=point.qz if point.qz is not None else 0.0,
            qw=point.qw if point.qw is not None else 1.0,
        )


def _extract_measurement_id(measurement_result: dict[str, Any] | None) -> Any:
    if measurement_result is None:
        return None
    for key in ("measurement_id", "id"):
        if key in measurement_result:
            return measurement_result[key]
    return None


def _extract_measurement_file_ref(measurement_result: dict[str, Any] | None) -> Any:
    if measurement_result is None:
        return None
    for key in ("file_ref", "file", "path"):
        if key in measurement_result:
            return measurement_result[key]
    return None
