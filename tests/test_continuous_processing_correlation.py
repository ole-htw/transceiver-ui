import numpy as np

from transceiver.helpers.continuous_processing import (
    _correlate_and_estimate_echo_aoa,
    _xcorr_fft_two_channel_batched,
)
from transceiver.helpers.correlation_utils import xcorr_fft


def test_batched_xcorr_matches_single_channel_reference():
    rng = np.random.default_rng(42)
    ch = rng.standard_normal((2, 513)) + 1j * rng.standard_normal((2, 513))
    txr = rng.standard_normal(321) + 1j * rng.standard_normal(321)

    cc_batched = _xcorr_fft_two_channel_batched(ch, txr)

    cc1_ref = xcorr_fft(ch[0], txr)
    cc2_ref = xcorr_fft(ch[1], txr)

    assert cc_batched.shape == (2, ch.shape[1] + txr.size - 1)
    assert np.allclose(cc_batched[0], cc1_ref, rtol=1e-6, atol=1e-6)
    assert np.allclose(cc_batched[1], cc2_ref, rtol=1e-6, atol=1e-6)


def test_correlate_and_estimate_echo_aoa_validation_mode():
    rng = np.random.default_rng(7)
    ch1 = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    ch2 = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    txr = rng.standard_normal(257) + 1j * rng.standard_normal(257)

    cc1, cc2 = _correlate_and_estimate_echo_aoa(ch1, ch2, txr, validate=True)

    assert np.allclose(cc1, xcorr_fft(ch1, txr), rtol=1e-6, atol=1e-6)
    assert np.allclose(cc2, xcorr_fft(ch2, txr), rtol=1e-6, atol=1e-6)
