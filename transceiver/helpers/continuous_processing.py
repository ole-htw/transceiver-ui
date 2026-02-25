from __future__ import annotations

import multiprocessing
import time

import numpy as np

from .correlation_utils import xcorr_fft as _xcorr_fft
from .path_cancellation import apply_path_cancellation


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
    candidate_mask = (
        (inner >= mag[:-2])
        & (inner >= mag[2:])
        & (inner >= thr)
    )
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


def _decimate_for_display(data: np.ndarray, max_points: int = 4096) -> np.ndarray:
    del max_points
    return np.asarray(data)


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
    validate: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
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
    return cc[0], cc[1]


def continuous_processing_worker(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    """Process continuous RX frames and emit preprocessed UI payloads."""
    cached_tx_path: str | None = None
    cached_tx_data = np.array([], dtype=np.complex64)

    while True:
        task = task_queue.get()
        if task is None:
            break

        started = time.monotonic()
        data = np.asarray(task.get("data", np.array([], dtype=np.complex64)))
        fs = float(task.get("fs", 1.0))
        tx_path = task.get("tx_path")
        trim_enabled = bool(task.get("trim_enabled", False))
        trim_start = float(task.get("trim_start", 0.0))
        trim_end = float(task.get("trim_end", 100.0))
        magnitude_enabled = bool(task.get("magnitude_enabled", False))
        rx_channel_view = str(task.get("rx_channel_view", "Kanal 1"))
        path_cancel_enabled = bool(task.get("path_cancel_enabled", False))

        if tx_path != cached_tx_path:
            cached_tx_path = tx_path
            cached_tx_data = np.array([], dtype=np.complex64)
            if tx_path:
                try:
                    raw = np.fromfile(tx_path, dtype=np.int16)
                    if raw.size % 2:
                        raw = raw[:-1]
                    raw = raw.reshape(-1, 2).astype(np.float32)
                    cached_tx_data = raw[:, 0] + 1j * raw[:, 1]
                except Exception:
                    cached_tx_data = np.array([], dtype=np.complex64)

        aoa_text = "AoA (ESPRIT): deaktiviert"
        echo_aoa_text = "Echo AoA: deaktiviert"

        plot_data = data
        if plot_data.ndim == 2 and plot_data.shape[0] >= 2:
            if rx_channel_view == "Kanal 2":
                plot_data = plot_data[1]
            elif rx_channel_view == "Differenz":
                plot_data = plot_data[0] - plot_data[1]
            else:
                plot_data = plot_data[0]
        if trim_enabled and plot_data.size:
            s = int(round(plot_data.size * trim_start / 100.0))
            e = int(round(plot_data.size * trim_end / 100.0))
            e = max(s + 1, min(plot_data.size, e))
            plot_data = plot_data[s:e]
        if magnitude_enabled:
            plot_data = np.abs(plot_data)

        if path_cancel_enabled and cached_tx_data.size and plot_data.size:
            try:
                plot_data, _ = apply_path_cancellation(plot_data, cached_tx_data)
            except Exception:
                pass

        plot_data = _decimate_for_display(np.asarray(plot_data))
        result_queue.put(
            {
                "frame_ts": float(task.get("frame_ts", started)),
                "fs": fs,
                "plot_data": plot_data,
                "aoa_text": aoa_text,
                "echo_aoa_text": echo_aoa_text,
                "aoa_series": None,
                "aoa_time": None,
                "processing_ms": (time.monotonic() - started) * 1000.0,
            }
        )
