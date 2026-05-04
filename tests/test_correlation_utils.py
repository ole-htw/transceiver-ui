import numpy as np

from transceiver.helpers.correlation_utils import (
    _suppress_nearby_candidates,
    classify_peak_group,
    classify_peak_group_from_mag,
    filter_peak_indices_to_period_group,
    find_local_maxima_around_peak,
    find_local_maxima_around_peak_from_mag,
    find_los_echo,
    find_los_echo_from_mag,
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


def test_suppress_nearby_candidates_keeps_strongest_per_cluster() -> None:
    mag = np.zeros(30, dtype=float)
    mag[10] = 0.6
    mag[11] = 0.9
    mag[20] = 0.7
    mag[22] = 0.8

    kept = _suppress_nearby_candidates([10, 11, 20, 22], mag, min_distance=2)

    assert kept == [11, 22]


def test_find_los_echo_from_mag_suppresses_echo_candidates_that_are_too_close() -> None:
    mag = np.zeros(220, dtype=float)
    mag[90] = 1.0
    mag[130] = 0.75
    mag[132] = 0.72

    los_idx, echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == 90
    assert echo_idx == 130


def test_local_maxima_with_repetition_period_only_scans_center_window() -> None:
    center = 150
    period = 80
    half = period // 2

    # Strictly monotonic baseline so only explicit spikes become local maxima.
    mag = np.linspace(0.0, 0.01, 300, dtype=float)

    # Peaks inside the permitted [center ± period/2] window.
    mag[center - 30] += 0.8
    mag[center] += 1.0
    mag[center + 25] += 0.75

    # Strong peaks outside the window that must never be returned.
    mag[center - half - 1] += 0.95
    mag[center + half + 1] += 0.9

    cc = mag.astype(np.complex128)
    maxima = find_local_maxima_around_peak(
        cc,
        center_idx=center,
        peaks_before=10,
        peaks_after=10,
        min_rel_height=0.0,
        repetition_period_samples=period,
    )

    assert maxima == [center - 30, center, center + 25]

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




def test_resolve_manual_los_idx_keeps_manual_selection_when_unconstrained() -> None:
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
        constrain_to_peak_group=False,
    )

    assert was_reset is False
    assert los_idx == 3
    assert manual_lags["los"] == -20

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
    expected_los_idx, expected_echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == expected_los_idx
    assert echo_idx == expected_echo_idx




def test_classify_peak_group_from_mag_with_los_threshold_filters_weak_echoes() -> None:
    mag = np.zeros(180, dtype=float)
    mag[90] = 1.0
    mag[110] = 0.25

    highest_idx, los_idx, echo_indices, group_indices = classify_peak_group_from_mag(
        mag,
        peaks_before=0,
        peaks_after=1,
        min_rel_height=0.3,
    )

    assert highest_idx == 90
    assert los_idx == 90
    assert echo_indices == []
    assert group_indices == [90]

def test_find_los_echo_from_mag_uses_stricter_los_threshold() -> None:
    mag = np.zeros(180, dtype=float)
    mag[90] = 1.0
    mag[110] = 0.25

    los_idx, echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == 90
    assert echo_idx == 110


def test_find_los_echo_from_mag_keeps_echo_above_los_threshold() -> None:
    mag = np.zeros(180, dtype=float)
    mag[90] = 1.0
    mag[110] = 0.35

    los_idx, echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == 90
    assert echo_idx == 110


def test_find_los_echo_from_mag_rejects_non_prominent_echo_in_noise() -> None:
    rng = np.random.default_rng(123)
    mag = np.abs(rng.normal(0.0, 0.04, 240))
    mag[120] = 1.0
    mag[150] = float(np.median(mag[142:159]) + 0.02)

    los_idx, echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == 120
    assert echo_idx is None


def test_find_los_echo_from_mag_ignores_echo_before_los_peak() -> None:
    mag = np.zeros(220, dtype=float)
    mag[90] = 0.45
    mag[120] = 1.0
    mag[150] = 0.4

    los_idx, echo_idx = find_los_echo_from_mag(mag)

    assert los_idx == 120
    assert echo_idx == 150


def test_from_mag_variants_match_cc_wrappers() -> None:
    n = 320
    mag = _sinc_lobe(n, 120, 1.0) + _sinc_lobe(n, 150, 0.72) + _sinc_lobe(n, 180, 0.66)
    cc = mag.astype(np.complex128)

    maxima_cc = find_local_maxima_around_peak(
        cc,
        center_idx=120,
        peaks_before=1,
        peaks_after=2,
        min_rel_height=0.05,
    )
    maxima_mag = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=120,
        peaks_before=1,
        peaks_after=2,
        min_rel_height=0.05,
    )
    assert maxima_mag == maxima_cc

    classified_cc = classify_peak_group(
        cc,
        peaks_before=1,
        peaks_after=2,
        min_rel_height=0.05,
    )
    classified_mag = classify_peak_group_from_mag(
        mag,
        peaks_before=1,
        peaks_after=2,
        min_rel_height=0.05,
    )
    assert classified_mag == classified_cc

    los_echo_cc = find_los_echo(cc)
    los_echo_mag = find_los_echo_from_mag(mag)
    assert los_echo_mag == los_echo_cc


def test_cc_wrapper_computes_abs_once(monkeypatch) -> None:
    n = 256
    mag = _sinc_lobe(n, 100, 1.0) + _sinc_lobe(n, 126, 0.7)
    cc = mag.astype(np.complex128)

    abs_calls = {"count": 0}
    original_abs = np.abs

    def counted_abs(value):
        abs_calls["count"] += 1
        return original_abs(value)

    monkeypatch.setattr(np, "abs", counted_abs)

    find_local_maxima_around_peak(cc, center_idx=100)

    assert abs_calls["count"] == 1


def _mag_with_right_flank_shoulder(length: int = 260) -> tuple[np.ndarray, int]:
    x = np.arange(length, dtype=float)
    mag = np.zeros(length, dtype=float)
    mag += 1.0 * np.exp(-0.5 * ((x - 90.0) / 2.3) ** 2)
    mag += 0.92 * np.exp(-0.5 * ((x - 118.0) / 2.7) ** 2)
    mag += 0.22 * np.exp(-0.5 * ((x - 155.0) / 2.0) ** 2)
    mag += 0.20 * np.exp(-0.5 * ((x - 175.0) / 2.1) ** 2)
    mag += 0.18 * np.exp(-0.5 * ((x - 195.0) / 2.2) ** 2)

    shoulder_center = 132
    shoulder = 0.04 * np.exp(-0.5 * ((x - shoulder_center) / 4.0) ** 2)
    tail = np.where(x >= 118.0, 0.34 * np.exp(-(x - 118.0) / 18.0), 0.0)
    mag += shoulder + tail
    return mag, shoulder_center


def test_include_shoulders_detects_descending_flank_hump() -> None:
    mag, shoulder_center = _mag_with_right_flank_shoulder()

    without_shoulders = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=4,
        min_rel_height=0.0,
        include_shoulders=False,
        repetition_period_samples=240,
    )
    with_shoulders = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=4,
        min_rel_height=0.0,
        include_shoulders=True,
        repetition_period_samples=240,
    )

    assert all(abs(idx - shoulder_center) > 2 for idx in without_shoulders)
    assert any(abs(idx - shoulder_center) <= 5 for idx in with_shoulders)


def test_include_shoulders_keeps_existing_strict_local_maxima() -> None:
    mag = np.zeros(220, dtype=float)
    mag[90] = 1.0
    mag[118] = 0.85
    mag[155] = 0.3
    mag[175] = 0.28

    maxima_no_shoulders = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=3,
        min_rel_height=0.0,
        include_shoulders=False,
        repetition_period_samples=240,
    )
    maxima_with_shoulders = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=3,
        min_rel_height=0.0,
        include_shoulders=True,
        repetition_period_samples=240,
    )

    for idx in maxima_no_shoulders:
        assert idx in maxima_with_shoulders


def test_shoulders_do_not_consume_peaks_after_budget() -> None:
    mag, shoulder_center = _mag_with_right_flank_shoulder()
    maxima = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=4,
        min_rel_height=0.0,
        include_shoulders=True,
        repetition_period_samples=240,
    )

    strict = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=4,
        min_rel_height=0.0,
        include_shoulders=False,
        repetition_period_samples=240,
    )
    assert strict[-1] in maxima
    assert any(abs(idx - shoulder_center) <= 5 for idx in maxima)


def test_shoulder_detection_rejects_small_noise_wiggles() -> None:
    x = np.arange(260, dtype=float)
    rng = np.random.default_rng(1234)
    mag = np.zeros_like(x, dtype=float)
    mag += 1.0 * np.exp(-0.5 * ((x - 100.0) / 2.6) ** 2)
    mag += np.where(x >= 100.0, 0.20 * np.exp(-(x - 100.0) / 20.0), 0.0)
    mag += 0.002 * rng.normal(size=x.size)

    maxima = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=100,
        peaks_before=0,
        peaks_after=0,
        min_rel_height=0.0,
        include_shoulders=True,
    )

    assert maxima == [100]


def test_shoulder_detection_respects_repetition_period_window() -> None:
    mag, _shoulder_center = _mag_with_right_flank_shoulder(length=340)
    outside_shoulder_center = 290
    x = np.arange(mag.size, dtype=float)
    mag += 0.13 * np.exp(-0.5 * ((x - outside_shoulder_center) / 3.0) ** 2)
    mag += np.where(x >= 275.0, 0.20 * np.exp(-(x - 275.0) / 14.0), 0.0)

    maxima = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=90,
        peaks_before=0,
        peaks_after=5,
        min_rel_height=0.0,
        include_shoulders=True,
        repetition_period_samples=140,
    )

    assert all(idx <= 160 for idx in maxima)
