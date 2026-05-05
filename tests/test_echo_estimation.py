import numpy as np
from transceiver.helpers.echo_estimation import EchoEstimatorConfig, estimate_echoes, simulate_multipath_received, zadoff_chu_sequence


def test_single_path():
    ref=zadoff_chu_sequence(5,31)
    rx=simulate_multipath_received(ref, delays_samples=[10], amplitudes=[1+0j], noise_power=0)
    r=estimate_echoes(rx,ref,EchoEstimatorConfig(sample_rate_hz=1.0, search_lag_min_samples=0))
    assert len(r.echoes)>=1


def test_noise_input_no_crash():
    ref=zadoff_chu_sequence(5,31)
    rng=np.random.default_rng(0)
    rx=rng.standard_normal(200)+1j*rng.standard_normal(200)
    r=estimate_echoes(rx,ref,EchoEstimatorConfig(sample_rate_hz=1.0, search_lag_min_samples=0))
    assert r is not None
