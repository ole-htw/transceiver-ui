from __future__ import annotations

import numpy as np

from .correlation_utils import xcorr_fft as _xcorr_fft


def _find_peaks_simple(
    mag: np.ndarray,
    rel_thresh: float = 0.2,
    min_dist: int = 200,
) -> list[int]:
    """Find local maxima in *mag* with a relative threshold and spacing."""
    if mag.size < 3:
        return []
    thr = float(rel_thresh) * float(np.max(mag))
    candidates = []
    for i in range(1, len(mag) - 1):
        if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: mag[i], reverse=True)
    block_radius = max(int(min_dist) - 1, 0)
    blocked = np.zeros(len(mag), dtype=bool)
    picked = []
    for i in candidates:
        if blocked[i]:
            continue
        picked.append(i)
        if block_radius > 0:
            start = max(i - block_radius, 0)
            end = min(i + block_radius + 1, len(mag))
            blocked[start:end] = True
    picked.sort()
    return picked


def _aoa_from_corr_peak(
    cc1: np.ndarray,
    cc2: np.ndarray,
    peak_index: int,
    antenna_spacing: float,
    wavelength: float,
    win: int = 50,
) -> tuple[float, float]:
    """Estimate AoA from correlation outputs around *peak_index*."""
    if antenna_spacing <= 0 or wavelength <= 0:
        return float("nan"), 0.0
    start = max(0, peak_index - win)
    end = min(len(cc1), peak_index + win + 1)
    w1 = cc1[start:end]
    w2 = cc2[start:end]
    mag = np.abs(w1) + np.abs(w2)
    if np.all(mag == 0):
        return float("nan"), 0.0

    z = np.sum(mag * (w2 * np.conj(w1)))
    denom = np.sum(mag * (np.abs(w1) * np.abs(w2))) + 1e-12
    coherence = float(np.abs(z) / denom)
    phi = float(np.angle(z))
    sin_theta = phi * wavelength / (2.0 * np.pi * antenna_spacing)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    theta = float(np.degrees(np.arcsin(sin_theta)))
    return theta, coherence


def _correlate_and_estimate_echo_aoa(
    rx_data: np.ndarray,
    tx_data: np.ndarray,
    antenna_spacing: float,
    wavelength: float,
    rel_thresh: float = 0.2,
    min_dist: int = 200,
    peak_win: int = 50,
) -> dict:
    """Cross-correlate per-channel RX with TX and estimate AoA per peak."""
    rx = np.asarray(rx_data)
    if rx.ndim != 2 or rx.shape[0] < 2:
        raise ValueError("Need two RX channels for echo AoA estimation.")

    ch1 = rx[0]
    ch2 = rx[1]
    n = min(len(ch1), len(ch2), len(tx_data))
    ch1 = ch1[:n]
    ch2 = ch2[:n]
    txr = tx_data[:n]

    cc1 = _xcorr_fft(ch1, txr)
    cc2 = _xcorr_fft(ch2, txr)
    lags = np.arange(-(len(txr) - 1), len(ch1))
    mag = np.abs(cc1) + np.abs(cc2)
    peaks = _find_peaks_simple(mag, rel_thresh=rel_thresh, min_dist=min_dist)

    results = []
    for p in peaks:
        theta, coh = _aoa_from_corr_peak(
            cc1,
            cc2,
            p,
            antenna_spacing=antenna_spacing,
            wavelength=wavelength,
            win=peak_win,
        )
        results.append(
            {
                "peak_index": int(p),
                "lag_samp": int(lags[p]),
                "strength": float(mag[p]),
                "theta_deg": float(theta),
                "coherence": float(coh),
            }
        )

    return {
        "lags": lags,
        "cc1": cc1,
        "cc2": cc2,
        "mag": mag,
        "peaks": peaks,
        "results": results,
    }

