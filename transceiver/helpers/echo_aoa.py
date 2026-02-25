from __future__ import annotations

import numpy as np

from .correlation_utils import xcorr_fft as _xcorr_fft


def _find_peaks_simple(
    mag: np.ndarray,
    rel_thresh: float = 0.2,
    min_dist: int = 100,
) -> list[int]:
    """Find local maxima in *mag* with relative threshold and spacing."""
    mag = np.asarray(mag)
    if mag.size < 3:
        return []

    thr = float(rel_thresh) * float(np.max(mag))
    inner = mag[1:-1]
    candidate_mask = (inner >= mag[:-2]) & (inner >= mag[2:]) & (inner >= thr)
    candidates = np.flatnonzero(candidate_mask) + 1
    if candidates.size == 0:
        return []

    min_dist = int(min_dist)
    if min_dist <= 1:
        return candidates.tolist()

    block_radius = min_dist - 1
    max_picks = max(1, int(np.ceil(mag.size / float(min_dist))))
    top_k = min(candidates.size, max(max_picks * 8, 64))

    if top_k < candidates.size:
        strong_sel = np.argpartition(mag[candidates], -top_k)[-top_k:]
        strong_candidates = candidates[strong_sel]
    else:
        strong_candidates = candidates

    strengths = mag[strong_candidates]
    order = np.argsort(-strengths, kind="stable")
    blocked = np.zeros(mag.size, dtype=bool)
    picked: list[int] = []

    for idx in strong_candidates[order]:
        i = int(idx)
        if blocked[i]:
            continue
        picked.append(i)
        start = max(i - block_radius, 0)
        end = min(i + block_radius + 1, mag.size)
        blocked[start:end] = True

    picked.sort()
    return picked


def _aoa_from_corr_peak(
    cc1: np.ndarray,
    cc2: np.ndarray,
    peak_idx: int,
    *,
    antenna_spacing_m: float = 0.5,
    wavelength_m: float = 1.0,
) -> float | None:
    """Estimate AoA in degrees from one correlation peak across two channels."""
    cc1 = np.asarray(cc1)
    cc2 = np.asarray(cc2)
    if cc1.size == 0 or cc2.size == 0:
        return None
    if peak_idx < 0 or peak_idx >= min(cc1.size, cc2.size):
        return None
    if antenna_spacing_m <= 0.0 or wavelength_m <= 0.0:
        return None

    phase_diff = float(np.angle(cc2[int(peak_idx)] * np.conj(cc1[int(peak_idx)])))
    sin_theta = (phase_diff * wavelength_m) / (2.0 * np.pi * antenna_spacing_m)
    if not np.isfinite(sin_theta):
        return None
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def _xcorr_fft_two_channel_batched(
    channels: np.ndarray,
    tx_reference: np.ndarray,
) -> np.ndarray:
    """Return FFT-based full cross-correlation for two channels in one batch."""
    if channels.ndim != 2 or channels.shape[0] != 2:
        raise ValueError("channels must be shaped as (2, N)")

    tx_reference = np.asarray(tx_reference)
    n_channels, len_a = channels.shape
    del n_channels
    len_b = tx_reference.size
    if len_a == 0 or len_b == 0:
        return np.zeros((2, max(0, len_a + len_b - 1)), dtype=np.complex128)

    n = len_a + len_b - 1
    nfft = 1 << (n - 1).bit_length()

    b_spec = np.fft.fft(tx_reference, nfft)
    a_spec = np.fft.fft(channels, nfft, axis=1)
    corr = np.fft.ifft(a_spec * np.conj(b_spec)[np.newaxis, :], axis=1)

    corr_tail = corr[:, -(len_b - 1) :] if len_b > 1 else corr[:, :0]
    return np.concatenate((corr_tail, corr[:, :len_a]), axis=1)


def _correlate_and_estimate_echo_aoa(
    ch1: np.ndarray,
    ch2: np.ndarray,
    txr: np.ndarray,
    *,
    return_debug_arrays: bool = False,
    validate: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Correlate channel 1 and 2 against ``txr`` using a single batched FFT pass."""
    ch = np.vstack((np.asarray(ch1), np.asarray(ch2)))
    cc = _xcorr_fft_two_channel_batched(ch, np.asarray(txr))
    if validate:
        cc1_ref = _xcorr_fft(np.asarray(ch1), np.asarray(txr))
        cc2_ref = _xcorr_fft(np.asarray(ch2), np.asarray(txr))
        if not np.allclose(cc[0], cc1_ref, rtol=rtol, atol=atol):
            raise AssertionError("Batched FFT correlation mismatch for channel 1")
        if not np.allclose(cc[1], cc2_ref, rtol=rtol, atol=atol):
            raise AssertionError("Batched FFT correlation mismatch for channel 2")

    result: dict[str, object] = {
        "results": (cc[0], cc[1]),
    }
    if return_debug_arrays:
        mag = np.abs(cc)
        result.update(
            {
                "cc1": cc[0],
                "cc2": cc[1],
                "mag": mag,
                "lags": np.arange(-np.asarray(txr).size + 1, np.asarray(ch1).size),
                "max_strength": float(np.max(mag)) if mag.size else 0.0,
            }
        )
    return result
