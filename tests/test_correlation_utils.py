import numpy as np
from transceiver.helpers.correlation_utils import xcorr_fft, autocorr_fft, correlation_lags, lag_overlap


def test_xcorr_fft_matches_numpy_correlate():
    a=np.array([1+1j,2-1j,0.5+0j])
    b=np.array([1-1j,0.5+0.2j])
    np.testing.assert_allclose(xcorr_fft(a,b), np.correlate(a,b,mode='full'))


def test_autocorr_and_lags_and_overlap():
    x=np.array([1+0j,2+0j,3+0j])
    ac=autocorr_fft(x)
    assert ac.shape[0]==5
    lags=correlation_lags(3,3)
    np.testing.assert_array_equal(lags, np.array([-2,-1,0,1,2]))
    assert lag_overlap(10,4,2)==(2,0,4)
