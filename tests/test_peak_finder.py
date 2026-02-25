import numpy as np

from transceiver.helpers.continuous_processing import _find_peaks_simple


def _find_peaks_simple_reference(
    mag: np.ndarray,
    rel_thresh: float = 0.2,
    min_dist: int = 100,
) -> list[int]:
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


def test_find_peaks_simple_plateau_behavior_matches_reference() -> None:
    mag = np.array([0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0])
    expected = _find_peaks_simple_reference(mag, rel_thresh=0.1, min_dist=1)
    assert _find_peaks_simple(mag, rel_thresh=0.1, min_dist=1) == expected


def test_find_peaks_simple_min_dist_regression() -> None:
    mag = np.array([0.0, 8.0, 0.0, 7.8, 0.0, 9.0, 0.0, 7.5, 0.0, 8.5, 0.0])
    expected = _find_peaks_simple_reference(mag, rel_thresh=0.1, min_dist=3)
    assert _find_peaks_simple(mag, rel_thresh=0.1, min_dist=3) == expected


def test_find_peaks_simple_randomized_matches_reference() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(100):
        mag = rng.random(200)
        rel_thresh = float(rng.uniform(0.0, 0.8))
        min_dist = int(rng.integers(1, 25))
        expected = _find_peaks_simple_reference(
            mag,
            rel_thresh=rel_thresh,
            min_dist=min_dist,
        )
        got = _find_peaks_simple(mag, rel_thresh=rel_thresh, min_dist=min_dist)
        assert got == expected
