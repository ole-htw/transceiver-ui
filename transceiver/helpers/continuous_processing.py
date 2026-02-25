from __future__ import annotations

import multiprocessing
import time

import numpy as np

from . import doa_esprit
from .echo_aoa import _correlate_and_estimate_echo_aoa
from .path_cancellation import apply_path_cancellation


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
