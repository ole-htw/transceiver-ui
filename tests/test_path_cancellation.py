import unittest

import numpy as np

from transceiver.helpers.correlation_utils import xcorr_fft
from transceiver.helpers.path_cancellation import apply_path_cancellation
from transceiver.helpers.tx_generator import generate_waveform


def _shift_with_zeros(x: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift < 0:
        out[:shift] = x[-shift:]
        return out
    if shift >= len(x):
        return out
    out[shift:] = x[: len(x) - shift]
    return out


def _synthesize_rx(
    tx_ref: np.ndarray,
    *,
    k0: int,
    k1: int | None = None,
    a0: complex = 1.0 + 0.0j,
    a1: complex = 0.0 + 0.0j,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rx = a0 * _shift_with_zeros(tx_ref, k0)
    if k1 is not None:
        rx = rx + a1 * _shift_with_zeros(tx_ref, k1)
    if noise_scale > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        noise = rng.normal(scale=noise_scale, size=tx_ref.shape) + 1j * rng.normal(
            scale=noise_scale, size=tx_ref.shape
        )
        rx = rx + noise.astype(np.complex64)
    return rx.astype(np.complex64)


def _corr_lags(n: int) -> np.ndarray:
    return np.arange(-n + 1, n)


class TestPathCancellation(unittest.TestCase):
    def setUp(self) -> None:
        self.tx_ref = generate_waveform(
            "zadoffchu",
            fs=1.0,
            f=1.0,
            N=1024,
            q=1,
            rrc_span=0,
            oversampling=1,
        )
        self.lags = _corr_lags(len(self.tx_ref))

    def _lag_index(self, lag: int) -> int:
        return int(np.where(self.lags == lag)[0][0])

    def test_two_path_cancellation_reduces_los(self) -> None:
        rng = np.random.default_rng(42)
        k0 = 80
        k1 = 300
        rx = _synthesize_rx(
            self.tx_ref,
            k0=k0,
            k1=k1,
            a0=1.0 + 0.0j,
            a1=0.3 - 0.1j,
            noise_scale=1e-3,
            rng=rng,
        )
        residual, _info = apply_path_cancellation(rx, self.tx_ref)
        corr = xcorr_fft(rx, self.tx_ref)
        corr2 = xcorr_fft(residual, self.tx_ref)

        k0_idx = self._lag_index(k0)
        k1_idx = self._lag_index(k1)
        self.assertLess(np.abs(corr2[k0_idx]), np.abs(corr[k0_idx]) / 10.0)

        peak_idx = int(np.argmax(np.abs(corr2)))
        self.assertLessEqual(abs(int(self.lags[peak_idx]) - k1), 1)
        ratio_before = np.abs(corr[k1_idx]) / (np.abs(corr[k0_idx]) + 1e-12)
        ratio_after = np.abs(corr2[k1_idx]) / (np.abs(corr2[k0_idx]) + 1e-12)
        self.assertGreater(ratio_after, ratio_before * 2.0)

    def test_close_in_echo_becomes_visible(self) -> None:
        k0 = 60
        k1 = 62
        rx = _synthesize_rx(
            self.tx_ref,
            k0=k0,
            k1=k1,
            a0=1.0 + 0.0j,
            a1=0.2 + 0.0j,
            noise_scale=5e-4,
            rng=np.random.default_rng(7),
        )
        residual, _info = apply_path_cancellation(rx, self.tx_ref)
        corr = xcorr_fft(rx, self.tx_ref)
        corr2 = xcorr_fft(residual, self.tx_ref)

        k0_idx = self._lag_index(k0)
        k1_idx = self._lag_index(k1)
        ratio_before = np.abs(corr[k1_idx]) / (np.abs(corr[k0_idx]) + 1e-12)
        ratio_after = np.abs(corr2[k1_idx]) / (np.abs(corr2[k0_idx]) + 1e-12)
        self.assertGreater(ratio_after, ratio_before * 2.0)

        peak_idx = int(np.argmax(np.abs(corr2)))
        self.assertLessEqual(abs(int(self.lags[peak_idx]) - k1), 1)

    def test_no_echo_reports_none(self) -> None:
        rx = _synthesize_rx(
            self.tx_ref,
            k0=90,
            k1=None,
            a0=1.0 + 0.0j,
            noise_scale=1e-3,
            rng=np.random.default_rng(123),
        )
        _residual, info = apply_path_cancellation(rx, self.tx_ref)
        self.assertIsNone(info["k1"])
        self.assertIn("Echo nach Pfad-Cancellation nicht detektierbar.", info["warning"])

    def test_bounds_case_warns_on_truncated_window(self) -> None:
        k0 = len(self.tx_ref) - 8
        rx = _synthesize_rx(self.tx_ref, k0=k0, k1=None, noise_scale=0.0)
        _residual, info = apply_path_cancellation(rx, self.tx_ref)
        self.assertTrue(info["applied"])
        self.assertIsNotNone(info["warning"])
        self.assertIn("Fenster gekürzt", info["warning"])


if __name__ == "__main__":
    unittest.main()
