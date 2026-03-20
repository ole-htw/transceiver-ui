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
