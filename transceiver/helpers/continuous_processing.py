from __future__ import annotations

import multiprocessing
import time
from multiprocessing import shared_memory

import numpy as np

from .echo_aoa import (
    _correlate_and_estimate_echo_aoa,
    _find_peaks_simple,
    _xcorr_fft_two_channel_batched,
)
from .path_cancellation import apply_path_cancellation
from .interpolation import apply_rx_interpolation


def _decimate_for_display(data: np.ndarray, max_points: int = 4096) -> np.ndarray:
    del max_points
    return np.asarray(data)


def _put_latest_result(result_queue: multiprocessing.Queue, payload: dict[str, object]) -> None:
    try:
        result_queue.put_nowait(payload)
        return
    except Exception:
        pass
    try:
        result_queue.get_nowait()
    except Exception:
        pass
    try:
        result_queue.put_nowait(payload)
    except Exception:
        pass


def continuous_processing_worker(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    input_slot_names: list[str] | None = None,
    input_slot_size: int = 0,
) -> None:
    """Process continuous RX frames and emit preprocessed UI payloads."""
    cached_tx_path: str | None = None
    cached_tx_data = np.array([], dtype=np.complex64)

    input_slots = [shared_memory.SharedMemory(name=name) for name in (input_slot_names or [])]
    use_shared_input = bool(input_slots) and input_slot_size > 0
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            started = time.monotonic()
            slot_id = int(task.get("slot_id", -1))
            nbytes = int(task.get("nbytes", 0))
            shape = tuple(int(v) for v in task.get("shape", (0,)))
            dtype = np.dtype(task.get("dtype", np.dtype(np.complex64).str))
            fs = float(task.get("fs", 1.0))
            tx_path = task.get("tx_path")
            trim_enabled = bool(task.get("trim_enabled", False))
            trim_start = float(task.get("trim_start", 0.0))
            trim_end = float(task.get("trim_end", 100.0))
            magnitude_enabled = bool(task.get("magnitude_enabled", False))
            rx_channel_view = str(task.get("rx_channel_view", "Kanal 1"))
            path_cancel_enabled = bool(task.get("path_cancel_enabled", False))
            interpolation_enabled = bool(task.get("interpolation_enabled", False))
            interpolation_method = str(task.get("interpolation_method", "interp1d"))
            interpolation_factor = str(task.get("interpolation_factor", "2"))

            if use_shared_input:
                if slot_id < 0 or slot_id >= len(input_slots) or nbytes <= 0 or nbytes > input_slot_size:
                    _put_latest_result(
                        result_queue,
                        {
                            "input_slot_id": slot_id,
                            "frame_ts": float(task.get("frame_ts", started)),
                            "fs": fs,
                            "plot_data": np.array([], dtype=np.complex64),
                            "aoa_text": "AoA (ESPRIT): deaktiviert",
                            "echo_aoa_text": "Echo AoA: deaktiviert",
                            "aoa_series": None,
                            "aoa_time": None,
                            "processing_ms": (time.monotonic() - started) * 1000.0,
                        },
                    )
                    continue

                slot_view = memoryview(input_slots[slot_id].buf)[:nbytes]
                data = np.frombuffer(slot_view, dtype=dtype)
                if shape:
                    data = data.reshape(shape)
            else:
                slot_view = None
                data = np.asarray(task.get("data", np.array([], dtype=np.complex64)))

            try:

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
                plot_data, interpolation_scale = apply_rx_interpolation(
                    plot_data,
                    enabled=interpolation_enabled,
                    method=interpolation_method,
                    factor_text=interpolation_factor,
                )
                if interpolation_scale > 1:
                    fs *= interpolation_scale
                if magnitude_enabled:
                    plot_data = np.abs(plot_data)

                if path_cancel_enabled and cached_tx_data.size and plot_data.size:
                    try:
                        plot_data, _ = apply_path_cancellation(plot_data, cached_tx_data)
                    except Exception:
                        pass

                plot_data = _decimate_for_display(np.asarray(plot_data))
                _put_latest_result(
                    result_queue,
                    {
                        "input_slot_id": slot_id,
                        "frame_ts": float(task.get("frame_ts", started)),
                        "fs": fs,
                        "plot_data": plot_data,
                        "aoa_text": aoa_text,
                        "echo_aoa_text": echo_aoa_text,
                        "aoa_series": None,
                        "aoa_time": None,
                        "processing_ms": (time.monotonic() - started) * 1000.0,
                    },
                )
            finally:
                if slot_view is not None:
                    slot_view.release()
    finally:
        for shm in input_slots:
            try:
                shm.close()
            except Exception:
                pass
