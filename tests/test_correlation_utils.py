import numpy as np

from transceiver.helpers.correlation_utils import (
    classify_peak_group,
    filter_peak_indices_to_period_group,
    find_local_maxima_around_peak,
    find_los_echo,
    resolve_manual_los_idx,
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


def test_resolve_manual_los_idx_resets_outdated_group_selection() -> None:
    lags = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], dtype=float)
    base_los_idx = 8
    highest_idx = 8
    period_samples = 40
    peak_group_indices = [7, 8, 9]
    manual_lags = {"los": -20, "echo": None}

    los_idx, was_reset = resolve_manual_los_idx(
        lags,
        base_los_idx,
        manual_lags,
        peak_group_indices=peak_group_indices,
        highest_idx=highest_idx,
        period_samples=period_samples,
    )

    assert was_reset is True
    assert los_idx == base_los_idx
    assert manual_lags["los"] is None


def test_classify_peak_group_returns_sorted_group_and_echoes() -> None:
    n = 220
    mag = _sinc_lobe(n, 80, 0.9) + _sinc_lobe(n, 120, 1.0) + _sinc_lobe(n, 150, 0.75)
    cc = mag.astype(np.complex128)

    highest_idx, los_idx, echo_indices, group_indices = classify_peak_group(
        cc,
        peaks_before=2,
        peaks_after=2,
        min_rel_height=0.05,
    )

    assert highest_idx == int(np.argmax(mag))
    assert group_indices == sorted(set(group_indices))
    assert los_idx == group_indices[0]
    assert echo_indices == group_indices[1:]
    assert highest_idx in group_indices


def test_find_los_echo_uses_classified_peak_group() -> None:
    n = 180
    mag = _sinc_lobe(n, 90, 1.0) + _sinc_lobe(n, 108, 0.7) + _sinc_lobe(n, 126, 0.65)
    cc = mag.astype(np.complex128)

    los_idx, echo_idx = find_los_echo(cc)
    _highest_idx, expected_los_idx, expected_echoes, _group = classify_peak_group(
        cc,
        peaks_before=0,
        peaks_after=1,
        min_rel_height=0.1,
    )

    assert los_idx == expected_los_idx
    assert echo_idx == (expected_echoes[0] if expected_echoes else None)
