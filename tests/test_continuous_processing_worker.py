import multiprocessing as mp
import threading

import numpy as np

from transceiver.helpers.continuous_processing import continuous_processing_worker


def test_continuous_worker_is_importable_from_non_main_module():
    assert continuous_processing_worker.__module__ == "transceiver.helpers.continuous_processing"


def test_continuous_worker_spawn_startup_and_shutdown():
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    proc = ctx.Process(target=continuous_processing_worker, args=(tasks, results), daemon=True)
    proc.start()
    tasks.put(None)
    proc.join(timeout=5)
    assert proc.exitcode == 0


def test_continuous_worker_keeps_full_resolution_for_plot_data():
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    proc = ctx.Process(target=continuous_processing_worker, args=(tasks, results), daemon=True)
    proc.start()

    sample_count = 8192
    channel_1 = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    channel_2 = np.linspace(1.0, 0.0, sample_count, dtype=np.float32)
    frame = np.vstack([channel_1, channel_2]).astype(np.complex64)

    tasks.put(
        {
            "data": frame,
            "fs": 1.0,
            "rx_channel_view": "Kanal 1",
        }
    )

    result = results.get(timeout=5)

    tasks.put(None)
    proc.join(timeout=5)

    assert proc.exitcode == 0
    assert result["plot_data"].shape[0] == sample_count


def test_continuous_worker_skips_tx_loading_outside_crosscorr_tab(monkeypatch):
    tasks: mp.Queue = mp.Queue()
    results: mp.Queue = mp.Queue()

    def _fail_fromfile(*_args, **_kwargs):
        raise AssertionError("np.fromfile should not be called for non-X-Corr tabs")

    monkeypatch.setattr(np, "fromfile", _fail_fromfile)

    worker = threading.Thread(
        target=continuous_processing_worker,
        args=(tasks, results),
        daemon=True,
    )
    worker.start()

    frame = np.ones((2, 128), dtype=np.complex64)
    tasks.put(
        {
            "data": frame,
            "fs": 1.0,
            "rx_channel_view": "Kanal 1",
            "path_cancel_enabled": True,
            "tx_path": "signals/tx/nonexistent.bin",
            "active_plot_tab": "Signal",
        }
    )

    result = results.get(timeout=5)
    tasks.put(None)
    worker.join(timeout=5)

    assert result["plot_data"].size == 128


def test_continuous_worker_keeps_latest_active_tab_for_follow_up_tasks(monkeypatch):
    tasks: mp.Queue = mp.Queue()
    results: mp.Queue = mp.Queue()

    def _fail_fromfile(*_args, **_kwargs):
        raise AssertionError("np.fromfile should not be called for non-X-Corr tabs")

    monkeypatch.setattr(np, "fromfile", _fail_fromfile)

    worker = threading.Thread(
        target=continuous_processing_worker,
        args=(tasks, results),
        daemon=True,
    )
    worker.start()

    frame = np.ones((2, 128), dtype=np.complex64)
    tasks.put(
        {
            "data": frame,
            "fs": 1.0,
            "rx_channel_view": "Kanal 1",
            "path_cancel_enabled": True,
            "tx_path": "signals/tx/nonexistent.bin",
            "active_plot_tab": "Signal",
        }
    )
    _ = results.get(timeout=5)

    tasks.put(
        {
            "data": frame,
            "fs": 1.0,
            "rx_channel_view": "Kanal 1",
            "path_cancel_enabled": True,
            "tx_path": "signals/tx/nonexistent.bin",
            "active_plot_tab": "Spectrum",
        }
    )

    spectrum_result = results.get(timeout=5)
    tasks.put(None)
    worker.join(timeout=5)

    assert spectrum_result["active_plot_tab"] == "Spectrum"
    assert spectrum_result["plot_data"].size == 128
