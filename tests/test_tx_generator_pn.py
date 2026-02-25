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
    )
    second = generate_waveform(
        "pseudo_noise",
        fs,
        0.0,
        N,
        pn_chip_rate=pn_chip_rate,
        pn_seed=pn_seed,
    )

    assert first.dtype == np.complex64
    assert len(first) == N
    assert np.array_equal(first, second)
    assert np.allclose(first.imag, 0.0)
    assert set(np.unique(first.real)).issubset({-1.0, 1.0})


def test_pseudo_noise_seed_changes_sequence() -> None:
    fs = 8e6
    N = 512

    first = generate_waveform("pseudo_noise", fs, 0.0, N, pn_seed=1)
    second = generate_waveform("pseudo_noise", fs, 0.0, N, pn_seed=2)

    assert len(first) == N
    assert len(second) == N
    assert not np.array_equal(first, second)
