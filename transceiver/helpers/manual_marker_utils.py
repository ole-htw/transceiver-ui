from __future__ import annotations

import numpy as np


def resolve_manual_marker_index(lags: np.ndarray, manual_lag: int | float | None) -> int | None:
    if manual_lag is None:
        return None
    lag_arr = np.asarray(lags)
    if lag_arr.ndim != 1 or lag_arr.size == 0:
        return None
    target = float(manual_lag)
    idx = int(np.argmin(np.abs(lag_arr.astype(float) - target)))
    if idx < 0 or idx >= lag_arr.size:
        return None
    return idx
