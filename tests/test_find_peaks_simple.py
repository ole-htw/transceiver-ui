import numpy as np

from transceiver.helpers.continuous_processing import _find_peaks_simple


def _legacy_find_peaks_simple(mag: np.ndarray, rel_thresh: float = 0.2, min_dist: int = 100) -> list[int]:
    if mag.size < 3:
        return []
    thr = float(rel_thresh) * float(np.max(mag))
    candidates = []
    for i in range(1, len(mag) - 1):
        if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: mag[i], reverse=True)
    min_dist = int(min_dist)
    block_radius = max(min_dist - 1, 0)
    blocked = np.zeros(len(mag), dtype=bool)
    picked = []
    for i in candidates:
        if blocked[i]:
            continue
        picked.append(i)
        if block_radius > 0:
            start = max(i - block_radius, 0)
            end = min(i + block_radius + 1, len(mag))
            blocked[start:end] = True
    picked.sort()
    return picked


def test_find_peaks_plateau_behavior_with_and_without_distance_suppression():
    mag = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], dtype=float)

    assert _find_peaks_simple(mag, rel_thresh=0.0, min_dist=1) == [2, 3]
    assert _find_peaks_simple(mag, rel_thresh=0.0, min_dist=2) == [2]


def test_find_peaks_distance_regression_matches_legacy_behavior():
    mag = np.array([0.0, 2.0, 8.0, 2.0, 0.0, 1.0, 6.0, 1.0, 0.0, 3.0, 7.0, 3.0, 0.0], dtype=float)

    expected = _legacy_find_peaks_simple(mag, rel_thresh=0.2, min_dist=4)
    assert _find_peaks_simple(mag, rel_thresh=0.2, min_dist=4) == expected


def test_find_peaks_matches_legacy_across_random_inputs():
    rng = np.random.default_rng(42)

    for n in [8, 16, 64, 257]:
        for min_dist in [1, 2, 5, 25]:
            for rel_thresh in [0.0, 0.15, 0.6]:
                mag = rng.random(n)
                expected = _legacy_find_peaks_simple(mag, rel_thresh=rel_thresh, min_dist=min_dist)
                observed = _find_peaks_simple(mag, rel_thresh=rel_thresh, min_dist=min_dist)
                assert observed == expected
