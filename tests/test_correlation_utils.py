import numpy as np

from transceiver.helpers.correlation_utils import find_los_echo


def test_find_los_echo_prefers_first_local_maximum_not_highest_peak():
    cc = np.array([0, 2, 0, 9, 0, 4, 0], dtype=float)
    los_idx, echo_idx = find_los_echo(cc)
    assert los_idx == 1
    assert echo_idx == 3


def test_find_los_echo_handles_empty_input():
    los_idx, echo_idx = find_los_echo(np.array([], dtype=float))
    assert los_idx is None
    assert echo_idx is None
