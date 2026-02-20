from __future__ import annotations

import multiprocessing
import time

import numpy as np

from . import doa_esprit
from .correlation_utils import xcorr_fft as _xcorr_fft
from .path_cancellation import apply_path_cancellation


def _find_peaks_simple(mag: np.ndarray, rel_thresh: float = 0.2, min_dist: int = 100) -> list[int]:
    if mag.size == 0:
        return []
    thr = float(rel_thresh) * float(np.max(mag))
    candidates = []
    for i in range(1, len(mag) - 1):
        if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: mag[i], reverse=True)
    min_dist = int(min_dist)
    block_radius = max(min_dist - 1, 0)
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
    if antenna_spacing <= 0 or wavelength <= 0:
        return float("nan"), 0.0
    start = max(0, peak_index - win)
    end = min(len(cc1), peak_index + win + 1)
    w1 = cc1[start:end]
    w2 = cc2[start:end]
    mag = np.abs(w1) + np.abs(w2)
    if np.all(mag == 0):
        return float("nan"), 0.0

    weight = mag
    z = np.sum(weight * (w2 * np.conj(w1)))
    denom = np.sum(weight * (np.abs(w1) * np.abs(w2))) + 1e-12
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


def _decimate_for_display(data: np.ndarray, max_points: int = 4096) -> np.ndarray:
    del max_points
    return np.asarray(data)


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
        antenna_spacing = task.get("antenna_spacing")
        wavelength = task.get("wavelength")
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

        aoa_text = "AoA (ESPRIT): --"
        echo_aoa_text = "Echo AoA: --"
        aoa_series = None
        aoa_time = None

        if data.ndim == 2 and data.shape[0] >= 2:
            aoa_raw = data[:2]
            if trim_enabled:
                s = int(round(aoa_raw.shape[1] * trim_start / 100.0))
                e = int(round(aoa_raw.shape[1] * trim_end / 100.0))
                e = max(s + 1, min(aoa_raw.shape[1], e))
                aoa_raw = aoa_raw[:, s:e]
            if antenna_spacing and wavelength:
                try:
                    aoa_angle = doa_esprit.estimate_aoa_esprit(
                        aoa_raw, float(antenna_spacing), float(wavelength)
                    )
                    if not np.isnan(aoa_angle):
                        aoa_text = f"AoA (ESPRIT): {aoa_angle:.1f}°"
                        aoa_time, aoa_series = doa_esprit.estimate_aoa_esprit_series(
                            aoa_raw,
                            float(antenna_spacing),
                            float(wavelength),
                        )
                except Exception as exc:
                    aoa_text = f"AoA (ESPRIT): Fehler ({exc})"

                if cached_tx_data.size > 0:
                    try:
                        echo_out = _correlate_and_estimate_echo_aoa(
                            aoa_raw,
                            cached_tx_data,
                            antenna_spacing=float(antenna_spacing),
                            wavelength=float(wavelength),
                        )
                        results = echo_out.get("results", [])
                        if results:
                            echo_aoa_text = "Echo AoA: " + ", ".join(
                                f"{item['lag_samp']}→{item['theta_deg']:.1f}°"
                                for item in results[:3]
                            )
                        else:
                            echo_aoa_text = "Echo AoA: keine Peaks"
                    except Exception:
                        echo_aoa_text = "Echo AoA: Berechnung fehlgeschlagen"

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
        aoa_series_plot = (
            None if aoa_series is None else _decimate_for_display(np.asarray(aoa_series))
        )
        aoa_time_plot = (
            None if aoa_time is None else _decimate_for_display(np.asarray(aoa_time))
        )

        result_queue.put(
            {
                "frame_ts": float(task.get("frame_ts", started)),
                "fs": fs,
                "plot_data": plot_data,
                "aoa_text": aoa_text,
                "echo_aoa_text": echo_aoa_text,
                "aoa_series": aoa_series_plot,
                "aoa_time": aoa_time_plot,
                "processing_ms": (time.monotonic() - started) * 1000.0,
            }
        )
