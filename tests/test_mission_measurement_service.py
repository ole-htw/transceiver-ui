from __future__ import annotations

import time

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
    service = MissionRxMeasurementService(
        app=_FakeApp(),
        on_status=lambda *_args: None,
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
        review_measurement=_review,
    )

    started = time.perf_counter()
    payload = service.trigger(_point_context())
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.045
    assert status_events == [("measurement", "running"), ("measurement", "succeeded")]
    assert timestamps["review_called"] >= timestamps["receive_end"]
    assert payload["file_ref"].endswith(".bin")
