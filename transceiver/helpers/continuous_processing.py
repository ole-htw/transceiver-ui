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

from .interpolation import _apply_rx_interpolation
from .path_cancellation import apply_path_cancellation


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


def _load_tx_samples(path: object) -> np.ndarray:
    if not path:
        return np.array([], dtype=np.complex64)
    try:
        raw = np.fromfile(str(path), dtype=np.int16)
    except Exception:
        return np.array([], dtype=np.complex64)
    if raw.size % 2:
        raw = raw[:-1]
    if raw.size == 0:
        return np.array([], dtype=np.complex64)
    raw = raw.reshape(-1, 2).astype(np.float32)
    return raw[:, 0] + 1j * raw[:, 1]


def _select_channel_and_trim(
    data: np.ndarray,
    *,
    rx_channel_view: str,
    trim_enabled: bool,
    trim_start: float,
    trim_end: float,
) -> np.ndarray:
    plot_data = np.asarray(data)
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
    return np.asarray(plot_data)


def _normalize_active_tab(tab: str) -> str:
    if tab == "X-Corr":
        return "X-Corr"
    if tab == "Spectrum":
        return "Spectrum"
    return "Signal"


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
            trim_enabled = bool(task.get("trim_enabled", False))
            trim_start = float(task.get("trim_start", 0.0))
            trim_end = float(task.get("trim_end", 100.0))
            magnitude_enabled = bool(task.get("magnitude_enabled", False))
            rx_channel_view = str(task.get("rx_channel_view", "Kanal 1"))
            path_cancel_enabled = bool(task.get("path_cancel_enabled", False))
            interpolation_enabled = bool(task.get("interpolation_enabled", False))
            interpolation_method = str(task.get("interpolation_method", "interp1d"))
            interpolation_factor = str(task.get("interpolation_factor", "2"))
            active_plot_tab = _normalize_active_tab(str(task.get("active_plot_tab", "Signal")))
            normalize_enabled = bool(
                task.get(
                    "normalize_enabled",
                    task.get("xcorr_normalized_enabled", False),
                )
            )

            if use_shared_input:
                if slot_id < 0 or slot_id >= len(input_slots) or nbytes <= 0 or nbytes > input_slot_size:
                    _put_latest_result(
                        result_queue,
                        {
                            "input_slot_id": slot_id,
                            "frame_ts": float(task.get("frame_ts", started)),
                            "fs": fs,
                            "plot_data": np.array([], dtype=np.complex64),
                            "plot_ref_data": np.array([], dtype=np.complex64),
                            "plot_ref_label": "TX",
                            "crosscorr_compare": None,
                            "path_cancel_info": None,
                            "active_plot_tab": active_plot_tab,
                            "normalize_enabled": normalize_enabled,
                            "xcorr_normalized_enabled": normalize_enabled,
                            "interpolation_enabled": interpolation_enabled,
                            "interpolation_applied": False,
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
                # Detach worker data from shared-memory buffer immediately so
                # shutdown can close segments without lingering exported
                # pointers from NumPy views.
                data = np.array(data, copy=True)
                slot_view.release()
                slot_view = None
            else:
                slot_view = None
                data = np.asarray(task.get("data", np.array([], dtype=np.complex64)))

            try:
                aoa_text = "AoA (ESPRIT): deaktiviert"
                echo_aoa_text = "Echo AoA: deaktiviert"

                plot_data = _select_channel_and_trim(
                    data,
                    rx_channel_view=rx_channel_view,
                    trim_enabled=trim_enabled,
                    trim_start=trim_start,
                    trim_end=trim_end,
                )
                ref_data = np.array([], dtype=np.complex64)
                ref_label = "TX"
                crosscorr_compare: np.ndarray | None = None
                path_cancel_info: dict[str, object] | None = None
                interpolation_applied = False

                # Signal + Spectrum: ausschließlich tab-spezifische Grundaufbereitung.
                # X-Corr: TX-Referenz, Path-Cancellation und Interpolation zentral hier.
                if active_plot_tab == "X-Corr":
                    tx_path = task.get("tx_path")
                    if tx_path != cached_tx_path:
                        cached_tx_path = str(tx_path) if tx_path else None
                        cached_tx_data = _load_tx_samples(tx_path)
                    ref_data = np.asarray(cached_tx_data)

                    if magnitude_enabled:
                        plot_data = np.abs(plot_data)
                        if ref_data.size:
                            ref_data = np.abs(ref_data)

                    if path_cancel_enabled and ref_data.size and plot_data.size:
                        crosscorr_compare = np.asarray(plot_data)
                        try:
                            plot_data, path_cancel_info = apply_path_cancellation(plot_data, ref_data)
                        except Exception:
                            path_cancel_info = None
                    if interpolation_enabled:
                        orig_plot_data = np.asarray(plot_data)
                        orig_ref_data = np.asarray(ref_data)
                        orig_crosscorr_compare = (
                            np.asarray(crosscorr_compare)
                            if crosscorr_compare is not None
                            else None
                        )
                        orig_fs = fs
                        try:
                            plot_data, fs = _apply_rx_interpolation(
                                plot_data,
                                fs=fs,
                                enabled=True,
                                method=interpolation_method,
                                factor_expr=interpolation_factor,
                            )
                            if ref_data.size:
                                ref_data, _ = _apply_rx_interpolation(
                                    ref_data,
                                    fs=fs,
                                    enabled=True,
                                    method=interpolation_method,
                                    factor_expr=interpolation_factor,
                                )
                            if crosscorr_compare is not None and crosscorr_compare.size:
                                crosscorr_compare, _ = _apply_rx_interpolation(
                                    crosscorr_compare,
                                    fs=fs,
                                    enabled=True,
                                    method=interpolation_method,
                                    factor_expr=interpolation_factor,
                                )
                        except Exception:
                            plot_data = orig_plot_data
                            ref_data = orig_ref_data
                            crosscorr_compare = orig_crosscorr_compare
                            fs = orig_fs
                            interpolation_applied = False
                        else:
                            interpolation_applied = True

                plot_data = _decimate_for_display(np.asarray(plot_data))
                if ref_data.size:
                    ref_data = _decimate_for_display(np.asarray(ref_data))
                if crosscorr_compare is not None and crosscorr_compare.size:
                    crosscorr_compare = _decimate_for_display(np.asarray(crosscorr_compare))

                _put_latest_result(
                    result_queue,
                    {
                        "input_slot_id": slot_id,
                        "frame_ts": float(task.get("frame_ts", started)),
                        "fs": fs,
                        "plot_data": plot_data,
                        "plot_ref_data": ref_data,
                        "plot_ref_label": ref_label,
                        "crosscorr_compare": crosscorr_compare,
                        "path_cancel_info": path_cancel_info,
                        "active_plot_tab": active_plot_tab,
                        "normalize_enabled": normalize_enabled,
                        "xcorr_normalized_enabled": normalize_enabled,
                        "interpolation_enabled": interpolation_enabled,
                        "interpolation_applied": interpolation_applied,
                        "interpolation_method": interpolation_method,
                        "interpolation_factor": interpolation_factor,
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
