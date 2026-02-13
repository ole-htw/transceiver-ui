import numpy as np

from transceiver.helpers.correlation_utils import find_local_maxima_around_peak


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
