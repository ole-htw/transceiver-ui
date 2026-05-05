from __future__ import annotations

import numpy as np


def _as_1d_complex(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr.astype(np.complex128, copy=False)


def xcorr_fft(data: np.ndarray, reference: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    data_c = _as_1d_complex(data, "data")
    ref_c = _as_1d_complex(reference, "reference")
    n = data_c.size + ref_c.size - 1
    nfft = 1 << (n - 1).bit_length()
    cc = np.fft.ifft(np.fft.fft(data_c, nfft) * np.conj(np.fft.fft(ref_c, nfft)))
    out = np.concatenate((cc[-(ref_c.size - 1):], cc[: data_c.size]))
    if normalize:
        energy = float(np.sum(np.abs(ref_c) ** 2))
        if energy <= 0.0:
            raise ValueError("reference energy is zero.")
        out = out / energy
    return out


def autocorr_fft(x: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    return xcorr_fft(x, x, normalize=normalize)


def correlation_lags(data_len: int, reference_len: int) -> np.ndarray:
    if data_len <= 0 or reference_len <= 0:
        raise ValueError("data_len and reference_len must be positive.")
    return np.arange(-(reference_len - 1), data_len, dtype=int)


def lag_overlap(data_len: int, reference_len: int, lag: int) -> tuple[int, int, int]:
    k = int(lag)
    r_start = max(0, k)
    s_start = max(0, -k)
    length = min(data_len - r_start, reference_len - s_start)
    return r_start, s_start, max(0, int(length))


def robust_baseline_sigma(x: np.ndarray) -> tuple[float, float]:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(y))
    mad = float(np.median(np.abs(y - baseline)))
    return baseline, 1.4826 * mad


def smooth_edge(x: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")
