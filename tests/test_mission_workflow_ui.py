from __future__ import annotations

from transceiver.mission_workflow_ui import MissionWorkflowWindow


def test_format_echo_delays_for_table_uses_structured_measurement_result_entries() -> None:
    text = MissionWorkflowWindow._format_echo_delays_for_table(
        [
            {"echo_index": 3, "delta_lag": 24, "distance_m": 36.0},
            {"echo_index": 5, "delta_lag": 42},
        ]
    )
    assert text == "E3: 24 (36.0m); E5: 42"


def test_yaw_conversion_uses_clockwise_degrees_in_ui() -> None:
    yaw_rad = MissionWorkflowWindow._yaw_cw_degrees_to_internal_radians("90")
    assert yaw_rad == -1.5707963267948966
    assert MissionWorkflowWindow._yaw_internal_radians_to_cw_degrees(yaw_rad) == 90.0
