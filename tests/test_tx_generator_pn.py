import numpy as np

from transceiver.helpers.tx_generator import generate_waveform


def test_pseudo_noise_deterministic_and_length() -> None:
    fs = 10e6
    N = 1024
    pn_chip_rate = 1e6
    pn_seed = 7

    first = generate_waveform(
        "pseudo_noise",
        fs,
        0.0,
        N,
        pn_chip_rate=pn_chip_rate,
        pn_seed=pn_seed,
        fd_zeroing_bandwidth=fs / 2,
    )
    second = generate_waveform(
        "pseudo_noise",
        fs,
        0.0,
        N,
        pn_chip_rate=pn_chip_rate,
        pn_seed=pn_seed,
        fd_zeroing_bandwidth=fs / 2,
    )

    assert first.dtype == np.complex64
    assert len(first) == N
    assert np.array_equal(first, second)
    assert np.allclose(first.imag, 0.0, atol=1e-6)
    assert np.all(np.isclose(np.abs(first.real), 1.0, atol=1e-6))


def test_pseudo_noise_seed_changes_sequence() -> None:
    fs = 8e6
    N = 512

    first = generate_waveform("pseudo_noise", fs, 0.0, N, pn_seed=1, fd_zeroing_bandwidth=fs / 2)
    second = generate_waveform("pseudo_noise", fs, 0.0, N, pn_seed=2, fd_zeroing_bandwidth=fs / 2)

    assert len(first) == N
    assert len(second) == N
    assert not np.array_equal(first, second)


def test_fd_zeroing_bandwidth_validation() -> None:
    fs = 10e6
    N = 256

    try:
        generate_waveform("sinus", fs, 1e6, N, fd_zeroing_bandwidth=None)
    except ValueError as exc:
        assert "Pflichtfeld" in str(exc)
    else:
        raise AssertionError("missing required fd_zeroing_bandwidth")

    try:
        generate_waveform("sinus", fs, 1e6, N, fd_zeroing_bandwidth=0.0)
    except ValueError as exc:
        assert "> 0" in str(exc)
    else:
        raise AssertionError("expected positive-bandwidth validation")

    try:
        generate_waveform("sinus", fs, 1e6, N, fd_zeroing_bandwidth=fs)
    except ValueError as exc:
        assert "Nyquist" in str(exc)
    else:
        raise AssertionError("expected Nyquist validation")
