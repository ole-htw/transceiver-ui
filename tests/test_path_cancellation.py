import numpy as np

from transceiver.helpers.correlation_utils import xcorr_fft
from transceiver.helpers.echo_estimation import zadoff_chu_sequence
from transceiver.helpers.path_cancellation import apply_path_cancellation


def _shift_with_zeros(x: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(x)
    if shift >= 0:
        out[shift:] = x[: len(x) - shift]
    return out


def test_path_cancellation_runs_and_reduces_main_peak():
    ref = zadoff_chu_sequence(5, 127).astype(np.complex64)
    rx = _shift_with_zeros(ref, 20) + 0.2 * _shift_with_zeros(ref, 40)
    residual, info = apply_path_cancellation(rx, ref)
    assert info["applied"] is True
    c0 = np.max(np.abs(xcorr_fft(rx, ref)))
    c1 = np.max(np.abs(xcorr_fft(residual, ref)))
    assert c1 < c0
