from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.navigation_adapter import NavigationPoint, TerminalNavigationState

ExecutorState = Literal["idle", "running", "paused", "stopping", "completed", "failed"]
OnPointError = Literal["continue", "stop"]
PointExecutionStatus = Literal["succeeded", "failed", "skipped"]


class MissionNavigator(Protocol):
    def navigate_to_point(
        self,
        point: NavigationPoint,
        *,
        timeout_s: float,
    ) -> TerminalNavigationState:
        ...


class MeasurementTrigger(Protocol):
    def __call__(self, point: MeasurementPoint) -> dict[str, Any]:
        ...


class ResultStore(Protocol):
    def __call__(self, payload: dict[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class MeasurementRunExecutorConfig:
    goal_reached_timeout_s: float = 120.0
    navigation_retry_attempts: int = 0
    on_point_error: OnPointError = "continue"


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
        trigger_measurement: MeasurementTrigger,
        persist_result: ResultStore,
        config: MeasurementRunExecutorConfig | None = None,
    ) -> None:
        self.mission = mission
        self.navigator = navigator
        self.trigger_measurement = trigger_measurement
        self.persist_result = persist_result
        self.config = config or MeasurementRunExecutorConfig()

        self._state: ExecutorState = "idle"
        self._state_lock = threading.RLock()
        self._pause_cond = threading.Condition(self._state_lock)
        self.records: list[PointExecutionRecord] = []

    @property
    def state(self) -> ExecutorState:
        with self._state_lock:
            return self._state

    def start(self) -> ExecutorState:
        with self._state_lock:
            if self._state not in {"idle", "completed", "failed"}:
                raise RuntimeError(f"Cannot start from state '{self._state}'")
            self._state = "running"

        self.records = []
        repeats = self.mission.repeat or 1
        for cycle in range(repeats):
            for point_index, point in enumerate(self.mission.points):
                if not self._wait_until_resumed_or_stopped():
                    return self._finalize_stop()

                if self.state == "stopping":
                    return self._finalize_stop()

                record = self._execute_point(
                    point=point,
                    point_index=point_index,
                    cycle=cycle,
                )
                self.records.append(record)

                if record.status == "failed" and self.config.on_point_error == "stop":
                    with self._state_lock:
                        self._state = "failed"
                    return self.state

                if self.mission.wait_after_arrival_s > 0 and self.state == "running":
                    time.sleep(self.mission.wait_after_arrival_s)

        with self._state_lock:
            if self._state == "stopping":
                return self._finalize_stop()
            self._state = "completed"
        return self.state

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
            self._state = "stopping"
            self._pause_cond.notify_all()

    def _finalize_stop(self) -> ExecutorState:
        with self._state_lock:
            self._state = "completed"
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
    ) -> PointExecutionRecord:
        nav_state: TerminalNavigationState | None = None
        attempts = self.config.navigation_retry_attempts + 1

        for attempt in range(1, attempts + 1):
            nav_state = self.navigator.navigate_to_point(
                self._to_navigation_point(point),
                timeout_s=self.config.goal_reached_timeout_s,
            )
            if nav_state == "succeeded":
                break

        if nav_state != "succeeded":
            return PointExecutionRecord(
                index=cycle * len(self.mission.points) + point_index,
                point_id=point.id,
                point_name=point.name,
                status="failed",
                navigation_state=nav_state,
                navigation_attempts=attempts,
                measurement_result=None,
                error=f"navigation_failed:{nav_state}",
            )

        try:
            measurement_result = self.trigger_measurement(point)
            payload = {
                "mission": self.mission.name,
                "cycle": cycle,
                "point": {
                    "id": point.id,
                    "name": point.name,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                    "yaw": point.yaw,
                    "qx": point.qx,
                    "qy": point.qy,
                    "qz": point.qz,
                    "qw": point.qw,
                    "notes": point.notes,
                    "measurement_profile": point.measurement_profile,
                },
                "measurement_result": measurement_result,
                "navigation": {
                    "state": nav_state,
                    "attempts": attempt,
                    "timeout_s": self.config.goal_reached_timeout_s,
                },
                "executor_state": self.state,
            }
            self.persist_result(payload)
        except Exception as exc:
            return PointExecutionRecord(
                index=cycle * len(self.mission.points) + point_index,
                point_id=point.id,
                point_name=point.name,
                status="failed",
                navigation_state=nav_state,
                navigation_attempts=attempt,
                measurement_result=None,
                error=str(exc),
            )

        return PointExecutionRecord(
            index=cycle * len(self.mission.points) + point_index,
            point_id=point.id,
            point_name=point.name,
            status="succeeded",
            navigation_state=nav_state,
            navigation_attempts=attempt,
            measurement_result=measurement_result,
            error=None,
        )

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
