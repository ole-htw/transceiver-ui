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


def _decimate_for_display(data: np.ndarray, max_points: int = 4096) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size <= max_points or max_points <= 0:
        return arr
    step = max(1, int(np.ceil(arr.size / max_points)))
    return arr[::step]


def continuous_processing_worker(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_id: int = 0,
) -> None:
    """Process continuous RX frames and emit preprocessed UI payloads."""
    cached_tx_path: str | None = None
    cached_tx_data = np.array([], dtype=np.complex64)
    input_shm_name: str | None = None
    input_shm: shared_memory.SharedMemory | None = None

    def _get_input_memory(name: str) -> shared_memory.SharedMemory | None:
        nonlocal input_shm_name, input_shm
        if input_shm_name == name and input_shm is not None:
            return input_shm
        if input_shm is not None:
            try:
                input_shm.close()
            except Exception:
                pass
            input_shm = None
            input_shm_name = None
        try:
            input_shm = shared_memory.SharedMemory(name=name)
            input_shm_name = name
        except Exception:
            return None
        return input_shm

    while True:
        task = task_queue.get()
        if task is None:
            break

        started = time.monotonic()
        seq_no = int(task.get("seq_no", -1))
        slot_id = int(task.get("slot_id", -1))
        data_shape = tuple(task.get("data_shape", ()))
        data_dtype = np.dtype(task.get("data_dtype", np.complex64))
        data_nbytes = int(task.get("data_nbytes", 0))
        shm_name = str(task.get("input_shm_name", ""))
        data = np.array([], dtype=np.complex64)
        if shm_name and data_shape and data_nbytes > 0:
            shm = _get_input_memory(shm_name)
            if shm is not None:
                slot_stride = int(task.get("slot_stride", data_nbytes))
                slot_offset = slot_stride * max(0, slot_id)
                if slot_offset + data_nbytes <= len(shm.buf):
                    data = np.ndarray(
                        data_shape,
                        dtype=data_dtype,
                        buffer=shm.buf,
                        offset=slot_offset,
                    )
        fs = float(task.get("fs", 1.0))
        tx_path = task.get("tx_path")
        trim_enabled = bool(task.get("trim_enabled", False))
        trim_start = float(task.get("trim_start", 0.0))
        trim_end = float(task.get("trim_end", 100.0))
        magnitude_enabled = bool(task.get("magnitude_enabled", False))
        rx_channel_view = str(task.get("rx_channel_view", "Kanal 1"))
        path_cancel_enabled = bool(task.get("path_cancel_enabled", False))
        heavy_every = max(1, int(task.get("heavy_every", 1)))
        adaptive_load_enabled = bool(task.get("adaptive_load_enabled", False))
        target_processing_ms = float(task.get("adaptive_target_processing_ms", 150.0))
        target_end_to_end_ms = float(task.get("adaptive_target_end_to_end_ms", 300.0))
        last_processing_ms = float(task.get("last_processing_ms", 0.0))
        last_end_to_end_ms = float(task.get("last_end_to_end_ms", 0.0))
        adaptive_factor = 1
        if adaptive_load_enabled:
            if last_processing_ms > target_processing_ms or last_end_to_end_ms > target_end_to_end_ms:
                adaptive_factor = 2
            if last_processing_ms > (target_processing_ms * 1.7) or last_end_to_end_ms > (target_end_to_end_ms * 1.7):
                adaptive_factor = 3
        effective_heavy_every = max(1, heavy_every * adaptive_factor)
        should_run_heavy = seq_no < 0 or (seq_no % effective_heavy_every == 0)

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

        if should_run_heavy and path_cancel_enabled and cached_tx_data.size and plot_data.size:
            try:
                plot_data, _ = apply_path_cancellation(plot_data, cached_tx_data)
            except Exception:
                pass

        plot_data = _decimate_for_display(np.asarray(plot_data))
        result_queue.put(
            {
                "frame_ts": float(task.get("frame_ts", started)),
                "seq_no": seq_no,
                "fs": fs,
                "plot_data": plot_data,
                "aoa_text": aoa_text,
                "echo_aoa_text": echo_aoa_text,
                "aoa_series": None,
                "aoa_time": None,
                "processing_ms": (time.monotonic() - started) * 1000.0,
                "worker_latency_ms": (time.monotonic() - float(task.get("frame_ts", started))) * 1000.0,
                "worker_id": worker_id,
                "effective_heavy_every": effective_heavy_every,
                "slot_id": slot_id,
            }
        )

    if input_shm is not None:
        try:
            input_shm.close()
        except Exception:
            pass
