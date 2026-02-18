import numpy as np

from transceiver.helpers.correlation_utils import (
    filter_peak_indices_to_period_group,
    find_local_maxima_around_peak,
)


def _sinc_lobe(length: int, center: int, amplitude: float) -> np.ndarray:
    x = np.arange(length, dtype=float) - float(center)
    return amplitude * np.abs(np.sinc(x / 6.0))


def test_local_maxima_are_limited_to_selected_los_lobe() -> None:
    n = 1500
    left_center = 250
    right_center = 1100

    mag = _sinc_lobe(n, left_center, 0.9) + _sinc_lobe(n, right_center, 1.0)
    cc = mag.astype(np.complex128)

    maxima = find_local_maxima_around_peak(
        cc,
        center_idx=right_center,
        peaks_before=4,
        peaks_after=4,
        min_rel_height=0.05,
    )

    assert right_center in maxima
    assert all(idx > 700 for idx in maxima)
    assert left_center not in maxima


def test_local_maxima_keep_visible_echoes_for_lower_secondary_peak() -> None:
    mag = np.zeros(200, dtype=float)
    mag[80] = 1.0
    mag[120] = 0.7
    cc = mag.astype(np.complex128)

    maxima = find_local_maxima_around_peak(
        cc,
        center_idx=80,
        peaks_before=0,
        peaks_after=2,
        min_rel_height=0.1,
    )

    assert maxima == [80, 120]


def test_local_maxima_do_not_include_adjacent_lower_group_when_period_known() -> None:
    n = 1200
    group_a_center = 420
    group_b_center = 720
    period = group_b_center - group_a_center

    mag = _sinc_lobe(n, group_a_center, 0.65) + _sinc_lobe(n, group_b_center, 1.0)
    cc = mag.astype(np.complex128)

    maxima = find_local_maxima_around_peak(
        cc,
        center_idx=group_b_center,
        peaks_before=6,
        peaks_after=6,
        min_rel_height=0.05,
        repetition_period_samples=period,
    )

    assert group_b_center in maxima
    assert all(abs(idx - group_b_center) <= period / 2 for idx in maxima)
    assert all(abs(idx - group_a_center) > period / 2 for idx in maxima)


def test_filter_peak_indices_to_period_group_removes_adjacent_repetition() -> None:
    lags = np.array([84000, 84500, 85000, 89500, 90000, 90500])
    peaks = [1, 2, 3, 4, 5]

    filtered = filter_peak_indices_to_period_group(
        lags, peaks, anchor_idx=1, period_samples=5000
    )

    assert filtered == [1, 2]


def test_filter_peak_indices_to_period_group_keeps_all_without_period() -> None:
    lags = np.array([10, 20, 30])
    peaks = [0, 1, 2]

    filtered = filter_peak_indices_to_period_group(
        lags, peaks, anchor_idx=1, period_samples=None
    )

    assert filtered == peaks
