from __future__ import annotations

from transceiver.measurement_mission import MeasurementPoint
from transceiver.measurement_run_executor import PointExecutionContext
from transceiver.mission_measurement_service import MissionRxMeasurementService


class _FakeApp:
    def receive_for_mission(self, *, output_file: str):
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
