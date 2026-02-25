from argparse import Namespace

import numpy as np
import pytest

from transceiver.helpers.tx_generator import (
    apply_frequency_domain_zeroing,
    apply_post_filter,
    generate_filename,
    generate_waveform,
)


def _tone(freq_hz: float, fs: float, n: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n)
    return (amp * np.exp(2j * np.pi * freq_hz * t / fs)).astype(np.complex64)


def test_apply_frequency_domain_zeroing_preserves_inband_and_suppresses_outband() -> None:
    fs = 8_000.0
    n = 4096
    bandwidth_hz = 2_000.0

    inband = _tone(500.0, fs, n, amp=1.0)
    outband = _tone(2_500.0, fs, n, amp=0.8)
    signal = inband + outband

    filtered = apply_frequency_domain_zeroing(signal, fs=fs, bandwidth_hz=bandwidth_hz)

    spec_before = np.fft.fft(signal)
    spec_after = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)

    in_idx = int(np.argmin(np.abs(freqs - 500.0)))
    out_idx = int(np.argmin(np.abs(freqs - 2_500.0)))

    in_ratio = np.abs(spec_after[in_idx]) / (np.abs(spec_before[in_idx]) + 1e-12)
    out_ratio = np.abs(spec_after[out_idx]) / (np.abs(spec_before[out_idx]) + 1e-12)

    assert in_ratio > 0.95
    assert out_ratio < 1e-3


def test_apply_frequency_domain_zeroing_preserves_length() -> None:
    fs = 1_000_000.0
    x = (np.random.default_rng(0).normal(size=777) + 1j * np.random.default_rng(1).normal(size=777)).astype(np.complex64)

    y = apply_frequency_domain_zeroing(x, fs=fs, bandwidth_hz=200_000.0)

    assert len(y) == len(x)


@pytest.mark.parametrize(
    "bandwidth_hz, expected",
    [
        (0.0, "bandwidth_hz muss > 0 sein."),
        (-1.0, "bandwidth_hz muss > 0 sein."),
        (1_100_000.0, "bandwidth_hz muss <= fs sein"),
    ],
)
def test_apply_frequency_domain_zeroing_parameter_validation(
    bandwidth_hz: float, expected: str
) -> None:
    x = np.ones(64, dtype=np.complex64)
    with pytest.raises(ValueError, match=expected):
        apply_frequency_domain_zeroing(x, fs=1_000_000.0, bandwidth_hz=bandwidth_hz)


def test_generate_like_path_filtering_without_rrc_or_oversampling_dependencies() -> None:
    fs = 2_000_000.0
    samples = 1024
    data = generate_waveform("zadoffchu", fs=fs, f=0.0, N=samples, q=7)

    filtered = apply_post_filter(
        data,
        fs=fs,
        filter_mode="fft_zeroing",
        filter_bandwidth_hz=200_000.0,
    )

    assert data.dtype == np.complex64
    assert filtered.dtype == np.complex64
    assert len(data) == samples
    assert len(filtered) == samples


def test_generate_filename_uses_bandwidth_token_and_bin_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedNow:
        def strftime(self, _fmt: str) -> str:
            return "20260101_120000"

    class _FixedDatetime:
        @staticmethod
        def now() -> _FixedNow:
            return _FixedNow()

    monkeypatch.setattr("transceiver.helpers.tx_generator.datetime", _FixedDatetime)

    args = Namespace(
        waveform="zadoffchu",
        f=0,
        f2=0,
        q=1,
        f0=0,
        f1=0,
        ofdm_nfft=64,
        ofdm_cp=16,
        ofdm_symbols=2,
        ofdm_short_repeats=0,
        pn_chip_rate=1e6,
        pn_seed=1,
        disable_filter=False,
        filter_bandwidth=2_500_000.0,
        samples=2048,
        output_dir="signals/tx",
    )

    name = generate_filename(args)

    assert name.suffix == ".bin"
    assert "_bw2p5M_" in name.name
    assert name.name.startswith("zadoffchu_q1_bw2p5M_Nsym2048_")
