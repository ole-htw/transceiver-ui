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


def test_best_fit_fraction_maximizes_scale_without_exceeding_target() -> None:
    fraction = MissionWorkflowWindow._best_fit_fraction(1.25, max_denominator=16)
    assert float(fraction) <= 1.25
    assert fraction == 5 / 4


def test_best_fit_fraction_returns_identity_for_invalid_scale() -> None:
    assert MissionWorkflowWindow._best_fit_fraction(0.0, max_denominator=16) == 1
