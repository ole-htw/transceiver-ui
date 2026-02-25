import numpy as np
import pytest

from transceiver.helpers.continuous_processing import _find_peaks_simple as _find_peaks_cp
from transceiver.helpers.echo_aoa import _find_peaks_simple as _find_peaks_shared


@pytest.mark.parametrize(
    "find_peaks_impl",
    [_find_peaks_cp, _find_peaks_shared],
)
def test_find_peaks_simple_matches_reference(find_peaks_impl) -> None:
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
        got = find_peaks_impl(mag, rel_thresh=rel_thresh, min_dist=min_dist)
        assert got == expected


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
